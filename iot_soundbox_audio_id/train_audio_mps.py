#!/usr/bin/env python3
"""Train an MPS-accelerated Audio CNN for IoT Soundbox EOL classification."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LOGGER = logging.getLogger("train_audio_mps")


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 44_100
    window_seconds: float = 1.0
    n_mels: int = 64
    n_fft: int = 1_024
    hop_length: int = 512
    fmin: float = 50.0
    fmax: float | None = None

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_seconds))


@dataclass(frozen=True)
class TrainConfig:
    manifest_path: Path = Path("data/processed/manifest.csv")
    model_path: Path = Path("models/iot_soundbox_audio_cnn.pth")
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    random_seed: int = 17
    num_workers: int = 0
    require_mps: bool = True
    patience: int = 12
    min_delta: float = 0.0


@dataclass(frozen=True)
class AudioExample:
    path: Path
    label_id: int
    class_name: str
    source_id: str


class AudioWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Loads one-second WAV windows and returns normalized mel-spectrogram tensors."""

    def __init__(
        self,
        examples: Sequence[AudioExample],
        feature_config: FeatureConfig,
        training: bool,
    ) -> None:
        self.examples = list(examples)
        self.feature_config = feature_config
        self.training = training

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        audio = load_window(example.path, self.feature_config)
        if self.training:
            audio = apply_dynamic_augmentations(audio, self.feature_config)
        mel = audio_to_log_mel(audio, self.feature_config)
        features = torch.from_numpy(mel).unsqueeze(0).float()
        label = torch.tensor(example.label_id, dtype=torch.long)
        return features, label


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class AudioCNN(nn.Module):
    """Compact CNN for one-channel mel-spectrogram classification."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout=0.05),
            ConvBlock(32, 64, dropout=0.10),
            ConvBlock(64, 128, dropout=0.15),
            ConvBlock(128, 192, dropout=0.20),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        x = self.pool(x)
        return self.classifier(x)


def parse_args() -> tuple[TrainConfig, FeatureConfig]:
    parser = argparse.ArgumentParser(description="Train IoT Soundbox AudioCNN on Apple MPS.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/iot_soundbox_audio_cnn.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Early-stop after this many epochs without validation accuracy improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum validation accuracy improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback if MPS is unavailable. MPS remains the default target.",
    )
    parser.add_argument("--sample-rate", type=int, default=44_100)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=1_024)
    parser.add_argument("--hop-length", type=int, default=512)
    args = parser.parse_args()

    train_config = TrainConfig(
        manifest_path=args.manifest,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        require_mps=not args.allow_cpu,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    feature_config = FeatureConfig(
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmax=args.sample_rate / 2.0,
    )
    return train_config, feature_config


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(require_mps: bool) -> torch.device:
    device = torch.device("mps")
    if torch.backends.mps.is_available():
        LOGGER.info("Using Apple Silicon accelerator: %s", device)
        return device
    if require_mps:
        raise RuntimeError(
            "MPS backend is unavailable. Run on an Apple Silicon PyTorch build or pass --allow-cpu."
        )
    LOGGER.warning("MPS unavailable; falling back to CPU because --allow-cpu was provided.")
    return torch.device("cpu")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_manifest(path: Path) -> list[AudioExample]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    examples: list[AudioExample] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "label_id", "class_name", "source_id"}
        missing = required.difference(reader.fieldnames or set())
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        for row in reader:
            audio_path = Path(row["path"])
            if not audio_path.exists():
                raise FileNotFoundError(f"Manifest audio path does not exist: {audio_path}")
            examples.append(
                AudioExample(
                    path=audio_path,
                    label_id=int(row["label_id"]),
                    class_name=row["class_name"],
                    source_id=row["source_id"],
                )
            )

    if not examples:
        raise ValueError(f"Manifest contains no training examples: {path}")
    return examples


def load_class_names(manifest_path: Path, examples: Sequence[AudioExample]) -> list[str]:
    class_map_path = manifest_path.parent / "class_map.json"
    if class_map_path.exists():
        with class_map_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        classes = sorted(payload["classes"], key=lambda item: int(item["id"]))
        return [str(item["name"]) for item in classes]

    id_to_name: dict[int, str] = {}
    for example in examples:
        id_to_name[example.label_id] = example.class_name
    return [id_to_name[index] for index in sorted(id_to_name)]


def split_train_val(
    examples: Sequence[AudioExample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[AudioExample], list[AudioExample]]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in the range (0.0, 0.5)")

    rng = random.Random(seed)
    grouped: dict[int, dict[str, list[AudioExample]]] = {}
    for example in examples:
        grouped.setdefault(example.label_id, {}).setdefault(example.source_id, []).append(example)

    train: list[AudioExample] = []
    val: list[AudioExample] = []

    for label_id, source_groups in grouped.items():
        groups = list(source_groups.values())
        rng.shuffle(groups)
        if len(groups) == 1:
            train.extend(groups[0])
            LOGGER.warning(
                "Label %d has one source file; validation will not contain this class.",
                label_id,
            )
            continue

        val_group_count = max(1, int(round(len(groups) * validation_fraction)))
        val_group_count = min(val_group_count, len(groups) - 1)
        for group in groups[:val_group_count]:
            val.extend(group)
        for group in groups[val_group_count:]:
            train.extend(group)

    rng.shuffle(train)
    rng.shuffle(val)
    if not train or not val:
        raise ValueError("Train/validation split is empty; add more raw source recordings.")
    return train, val


def load_window(path: Path, feature_config: FeatureConfig) -> np.ndarray:
    try:
        audio, _ = librosa.load(path, sr=feature_config.sample_rate, mono=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load training audio {path}: {exc}") from exc

    if audio.size == 0:
        raise ValueError(f"Training audio is empty: {path}")
    audio = librosa.util.fix_length(
        audio.astype(np.float32, copy=False),
        size=feature_config.window_samples,
    )
    return audio.astype(np.float32, copy=False)


def apply_dynamic_augmentations(audio: np.ndarray, feature_config: FeatureConfig) -> np.ndarray:
    """Apply blueprint dynamic training augmentation in waveform space."""

    augmented = audio.astype(np.float32, copy=True)

    gain_db = float(np.random.uniform(-3.0, 3.0))
    augmented *= 10.0 ** (gain_db / 20.0)

    if np.random.random() < 0.60:
        n_steps = float(np.random.uniform(-0.5, 0.5))
        augmented = librosa.effects.pitch_shift(
            augmented,
            sr=feature_config.sample_rate,
            n_steps=n_steps,
        ).astype(np.float32, copy=False)

    if np.random.random() < 0.60:
        stretch_rate = float(np.random.uniform(0.95, 1.05))
        augmented = librosa.effects.time_stretch(augmented, rate=stretch_rate).astype(
            np.float32,
            copy=False,
        )

    augmented = librosa.util.fix_length(augmented, size=feature_config.window_samples)
    return np.clip(augmented, -1.0, 1.0).astype(np.float32, copy=False)


def audio_to_log_mel(audio: np.ndarray, feature_config: FeatureConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=feature_config.sample_rate,
        n_fft=feature_config.n_fft,
        hop_length=feature_config.hop_length,
        n_mels=feature_config.n_mels,
        fmin=feature_config.fmin,
        fmax=feature_config.fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    normalized = (log_mel + 80.0) / 80.0
    return normalized.astype(np.float32, copy=False)


def class_weights(examples: Sequence[AudioExample], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for example in examples:
        if example.label_id >= num_classes:
            raise ValueError(
                f"Label id {example.label_id} exceeds model class count {num_classes}."
            )
        counts[example.label_id] += 1.0

    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (float(nonzero.sum()) * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_dataloaders(
    train_examples: Sequence[AudioExample],
    val_examples: Sequence[AudioExample],
    feature_config: FeatureConfig,
    train_config: TrainConfig,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    train_dataset = AudioWindowDataset(train_examples, feature_config, training=True)
    val_dataset = AudioWindowDataset(val_examples, feature_config, training=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("DataLoader produced zero examples")

    return total_loss / total_examples, total_correct / total_examples


def save_checkpoint(
    model: nn.Module,
    path: Path,
    class_names: Sequence[str],
    feature_config: FeatureConfig,
    train_config: TrainConfig,
    epoch: int,
    val_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_name": "AudioCNN",
        "class_names": list(class_names),
        "feature_config": asdict(feature_config),
        "train_config": {
            **asdict(train_config),
            "manifest_path": str(train_config.manifest_path),
            "model_path": str(train_config.model_path),
        },
        "epoch": epoch,
        "val_accuracy": val_accuracy,
    }
    torch.save(payload, path)


def train(train_config: TrainConfig, feature_config: FeatureConfig) -> None:
    set_reproducibility(train_config.random_seed)
    device = resolve_device(train_config.require_mps)

    examples = read_manifest(train_config.manifest_path)
    class_names = load_class_names(train_config.manifest_path, examples)
    train_examples, val_examples = split_train_val(
        examples,
        train_config.validation_fraction,
        train_config.random_seed,
    )

    LOGGER.info("Training examples: %d", len(train_examples))
    LOGGER.info("Validation examples: %d", len(val_examples))
    LOGGER.info("Classes: %s", ", ".join(class_names))

    train_loader, val_loader = make_dataloaders(
        train_examples,
        val_examples,
        feature_config,
        train_config,
    )

    model = AudioCNN(num_classes=len(class_names)).to(device)
    weights = class_weights(train_examples, len(class_names), device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_config.epochs),
    )

    best_val_accuracy = -math.inf
    epochs_without_improvement = 0
    for epoch in range(1, train_config.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        with torch.no_grad():
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        LOGGER.info(
            "Epoch %03d/%03d | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
            epoch,
            train_config.epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        improved = val_accuracy > best_val_accuracy + train_config.min_delta
        if improved:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                path=train_config.model_path,
                class_names=class_names,
                feature_config=feature_config,
                train_config=train_config,
                epoch=epoch,
                val_accuracy=val_accuracy,
            )
            LOGGER.info("Saved best checkpoint: %s", train_config.model_path)
        else:
            epochs_without_improvement += 1
            if train_config.patience > 0:
                LOGGER.info(
                    "No validation improvement for %d/%d epoch(s).",
                    epochs_without_improvement,
                    train_config.patience,
                )

        if train_config.patience > 0 and epochs_without_improvement >= train_config.patience:
            LOGGER.info("Early stopping triggered at epoch %d.", epoch)
            break

    LOGGER.info("Best validation accuracy: %.4f", best_val_accuracy)


def main() -> int:
    configure_logging()
    train_config, feature_config = parse_args()
    try:
        train(train_config, feature_config)
    except Exception:
        LOGGER.exception("Training failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
