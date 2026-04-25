#!/usr/bin/env python3
"""Realtime IoT Soundbox audio inference engine.

The engine captures a continuous 44.1 kHz PyAudio stream, maintains a rolling
one-second buffer, and evaluates the trained AudioCNN every 0.5 seconds.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import librosa
import numpy as np
import torch
from torch import nn


LOGGER = logging.getLogger("realtime_inference")
FUNCTIONAL_PREFIXES = ("01_", "02_", "03_", "04_", "05_", "06_")
DISTORTION_PREFIX = "09_"
DEFAULT_INPUT_DEVICE_NAME = "MacBook Pro Microphone"
PROTOTYPE_CLASS_THRESHOLDS: dict[str, float] = {
    "02_power_off": 0.50,
    "03_beep": 0.70,
    "09_distortion": 0.90,
}


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
class RuntimeConfig:
    model_path: Path
    profile: str = "prototype"
    threshold: float = 0.90
    stream_sample_rate: int = 48_000
    channels: int = 1
    cooldown_seconds: float = 0.75
    input_device_index: int | None = None
    input_device_name: str | None = DEFAULT_INPUT_DEVICE_NAME
    debug_top_k: int = 0
    debug_interval_seconds: float = 0.5
    class_thresholds: dict[str, float] = field(default_factory=dict)
    expected_class: str | None = None
    list_devices: bool = False

    @property
    def stream_window_samples(self) -> int:
        return int(round(self.stream_sample_rate * 1.0))

    @property
    def stream_hop_samples(self) -> int:
        return int(round(self.stream_sample_rate * 0.5))


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


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Run realtime IoT Soundbox inference.")
    parser.add_argument("--model-path", type=Path, default=Path("models/iot_soundbox_audio_cnn.pth"))
    parser.add_argument(
        "--profile",
        choices=("prototype", "strict"),
        default="prototype",
        help="Threshold preset. Prototype uses tuned thresholds for the current small model.",
    )
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--stream-sample-rate",
        type=int,
        default=48_000,
        help="Microphone stream rate. MacBook Pro microphones normally use 48000 Hz.",
    )
    parser.add_argument("--cooldown-seconds", type=float, default=0.75)
    parser.add_argument("--input-device-index", type=int, default=None)
    parser.add_argument(
        "--input-device-name",
        type=str,
        default=DEFAULT_INPUT_DEVICE_NAME,
        help="Preferred input device name substring. Ignored when --input-device-index is set.",
    )
    parser.add_argument(
        "--debug-top-k",
        type=int,
        default=0,
        help="Print the top K class probabilities for every debug interval. Use 3 for live diagnosis.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Shortcut for --debug-top-k 3.",
    )
    parser.add_argument(
        "--debug-interval-seconds",
        type=float,
        default=0.5,
        help="Minimum time between debug top-k lines.",
    )
    parser.add_argument(
        "--class-thresholds",
        type=str,
        default="",
        help=(
            "Comma-separated per-class thresholds, for example "
            "'02_power_off=0.50,09_distortion=0.90'."
        ),
    )
    parser.add_argument(
        "--expected-class",
        type=str,
        default=None,
        help="Only emit functional EVENT lines for this class. Distortion failures still emit.",
    )
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in the range (0, 1)")
    debug_top_k = 3 if args.debug else args.debug_top_k
    if debug_top_k < 0:
        raise ValueError("--debug-top-k must be zero or positive")
    if args.debug_interval_seconds <= 0:
        raise ValueError("--debug-interval-seconds must be positive")
    class_thresholds = default_class_thresholds(args.profile)
    class_thresholds.update(parse_class_thresholds(args.class_thresholds))

    return RuntimeConfig(
        model_path=args.model_path,
        profile=args.profile,
        threshold=args.threshold,
        stream_sample_rate=args.stream_sample_rate,
        cooldown_seconds=args.cooldown_seconds,
        input_device_index=args.input_device_index,
        input_device_name=args.input_device_name,
        debug_top_k=debug_top_k,
        debug_interval_seconds=args.debug_interval_seconds,
        class_thresholds=class_thresholds,
        expected_class=args.expected_class,
        list_devices=args.list_devices,
    )


def default_class_thresholds(profile: str) -> dict[str, float]:
    if profile == "prototype":
        return dict(PROTOTYPE_CLASS_THRESHOLDS)
    if profile == "strict":
        return {}
    raise ValueError(f"Unsupported inference profile: {profile}")


def parse_class_thresholds(raw_thresholds: str) -> dict[str, float]:
    """Parse exact per-class confidence thresholds from CLI text."""

    thresholds: dict[str, float] = {}
    if not raw_thresholds.strip():
        return thresholds

    for item in raw_thresholds.split(","):
        if not item.strip():
            continue
        class_name, separator, raw_value = item.partition("=")
        if not separator:
            raise ValueError(
                "Each --class-thresholds item must use class_name=value format."
            )
        class_name = class_name.strip()
        if not class_name:
            raise ValueError("--class-thresholds contains an empty class name")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid threshold for {class_name!r}: {raw_value!r}") from exc
        if not 0.0 < value < 1.0:
            raise ValueError(f"Threshold for {class_name!r} must be in the range (0, 1)")
        thresholds[class_name] = value
    return thresholds


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def import_pyaudio() -> Any:
    try:
        import pyaudio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PyAudio is required for realtime inference. Install it with "
            "`brew install portaudio` and `python -m pip install pyaudio`."
        ) from exc
    return pyaudio


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        LOGGER.info("Using Apple Silicon accelerator: %s", device)
        return device
    LOGGER.warning("MPS unavailable; using CPU for realtime inference.")
    return torch.device("cpu")


def feature_config_from_checkpoint(payload: dict[str, Any]) -> FeatureConfig:
    raw_config = dict(payload.get("feature_config", {}))
    return FeatureConfig(
        sample_rate=int(raw_config.get("sample_rate", 44_100)),
        window_seconds=float(raw_config.get("window_seconds", 1.0)),
        n_mels=int(raw_config.get("n_mels", 64)),
        n_fft=int(raw_config.get("n_fft", 1_024)),
        hop_length=int(raw_config.get("hop_length", 512)),
        fmin=float(raw_config.get("fmin", 50.0)),
        fmax=raw_config.get("fmax", 22_050.0),
    )


def load_model(
    model_path: Path,
    device: torch.device,
) -> tuple[AudioCNN, list[str], FeatureConfig]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    payload = torch.load(model_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint format: {model_path}")

    class_names = [str(name) for name in payload.get("class_names", [])]
    if not class_names:
        raise ValueError("Checkpoint is missing class_names")

    model = AudioCNN(num_classes=len(class_names))
    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint is missing model_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    feature_config = feature_config_from_checkpoint(payload)
    LOGGER.info("Loaded checkpoint: %s", model_path)
    LOGGER.info("Classes: %s", ", ".join(class_names))
    return model, class_names, feature_config


def list_input_devices(pyaudio_module: Any) -> None:
    audio = pyaudio_module.PyAudio()
    try:
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(
                    f"{index}: {info.get('name')} "
                    f"(inputs={info.get('maxInputChannels')}, "
                    f"default_sr={info.get('defaultSampleRate')})"
                )
    finally:
        audio.terminate()


def resolve_input_device_index(
    audio: Any,
    requested_index: int | None,
    requested_name: str | None,
) -> int | None:
    """Resolve the input device, preferring the MacBook microphone by default."""

    if requested_index is not None:
        info = audio.get_device_info_by_index(requested_index)
        LOGGER.info(
            "Using input device %d: %s",
            requested_index,
            info.get("name", "unknown"),
        )
        return requested_index

    if requested_name:
        normalized_requested_name = requested_name.casefold()
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            device_name = str(info.get("name", ""))
            if normalized_requested_name in device_name.casefold():
                LOGGER.info("Using input device %d: %s", index, device_name)
                return index
        LOGGER.warning(
            "Input device containing %r was not found; using PyAudio default input.",
            requested_name,
        )

    default_info = audio.get_default_input_device_info()
    LOGGER.info(
        "Using default input device %d: %s",
        int(default_info.get("index", -1)),
        default_info.get("name", "unknown"),
    )
    return None


def pcm16_to_float32(data: bytes, expected_samples: int) -> np.ndarray:
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.shape[0] == expected_samples:
        return samples
    if samples.shape[0] > expected_samples:
        return samples[:expected_samples]
    padded = np.zeros(expected_samples, dtype=np.float32)
    padded[: samples.shape[0]] = samples
    return padded


def prepare_model_audio(
    stream_window: np.ndarray,
    stream_sample_rate: int,
    feature_config: FeatureConfig,
) -> np.ndarray:
    audio = stream_window.astype(np.float32, copy=False)
    if stream_sample_rate != feature_config.sample_rate:
        audio = librosa.resample(
            y=audio,
            orig_sr=stream_sample_rate,
            target_sr=feature_config.sample_rate,
        ).astype(np.float32, copy=False)
    audio = librosa.util.fix_length(audio, size=feature_config.window_samples)
    return audio.astype(np.float32, copy=False)


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


@torch.no_grad()
def predict(
    model: AudioCNN,
    stream_window: np.ndarray,
    runtime_config: RuntimeConfig,
    feature_config: FeatureConfig,
    device: torch.device,
) -> tuple[int, float, np.ndarray]:
    model_audio = prepare_model_audio(
        stream_window,
        runtime_config.stream_sample_rate,
        feature_config,
    )
    mel = audio_to_log_mel(model_audio, feature_config)
    tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    return predicted_index, confidence, probabilities


def debug_prediction_line(
    class_names: Sequence[str],
    probabilities: np.ndarray,
    top_k: int,
) -> str:
    top_k = max(1, min(top_k, len(class_names)))
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    parts = [
        f"{class_names[int(index)]}={float(probabilities[int(index)]):.3f}"
        for index in top_indices
    ]
    return " | ".join(parts)


def should_emit(
    class_name: str,
    now_seconds: float,
    last_emit_by_class: dict[str, float],
    cooldown_seconds: float,
) -> bool:
    last_emit = last_emit_by_class.get(class_name, 0.0)
    if now_seconds - last_emit < cooldown_seconds:
        return False
    last_emit_by_class[class_name] = now_seconds
    return True


def handle_prediction(
    class_names: Sequence[str],
    predicted_index: int,
    confidence: float,
    threshold: float,
    class_thresholds: Mapping[str, float],
    expected_class: str | None,
    last_emit_by_class: dict[str, float],
    cooldown_seconds: float,
) -> None:
    class_name = class_names[predicted_index]
    required_confidence = class_thresholds.get(class_name, threshold)
    now_seconds = time.monotonic()
    timestamp = datetime.now().isoformat(timespec="milliseconds")

    if class_name.startswith(DISTORTION_PREFIX):
        if confidence >= required_confidence and should_emit(
            class_name,
            now_seconds,
            last_emit_by_class,
            cooldown_seconds,
        ):
            print(f"{timestamp} | FAILURE | class={class_name} | confidence={confidence:.3f}")
        return

    if expected_class is not None and class_name != expected_class:
        return

    if class_name.startswith(FUNCTIONAL_PREFIXES) and confidence >= required_confidence:
        if should_emit(class_name, now_seconds, last_emit_by_class, cooldown_seconds):
            print(f"{timestamp} | EVENT   | class={class_name} | confidence={confidence:.3f}")


def run_stream(
    runtime_config: RuntimeConfig,
    model: AudioCNN,
    class_names: Sequence[str],
    feature_config: FeatureConfig,
    device: torch.device,
    pyaudio_module: Any,
) -> None:
    audio = pyaudio_module.PyAudio()
    stream = None
    rolling_buffer = np.zeros(runtime_config.stream_window_samples, dtype=np.float32)
    samples_seen = 0
    last_emit_by_class: dict[str, float] = {}
    last_debug_emit = 0.0

    try:
        input_device_index = resolve_input_device_index(
            audio,
            runtime_config.input_device_index,
            runtime_config.input_device_name,
        )
        stream = audio.open(
            format=pyaudio_module.paInt16,
            channels=runtime_config.channels,
            rate=runtime_config.stream_sample_rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=runtime_config.stream_hop_samples,
        )
        LOGGER.info(
            "Listening at %d Hz. Model audio is resampled to %d Hz. Press Ctrl+C to stop.",
            runtime_config.stream_sample_rate,
            feature_config.sample_rate,
        )
        if runtime_config.class_thresholds:
            LOGGER.info(
                "Inference profile: %s | default threshold %.2f | class thresholds %s",
                runtime_config.profile,
                runtime_config.threshold,
                ", ".join(
                    f"{class_name}={value:.2f}"
                    for class_name, value in sorted(runtime_config.class_thresholds.items())
                ),
            )
        else:
            LOGGER.info(
                "Inference profile: %s | default threshold %.2f",
                runtime_config.profile,
                runtime_config.threshold,
            )

        while True:
            data = stream.read(
                runtime_config.stream_hop_samples,
                exception_on_overflow=False,
            )
            chunk = pcm16_to_float32(data, runtime_config.stream_hop_samples)
            rolling_buffer[:-chunk.shape[0]] = rolling_buffer[chunk.shape[0] :]
            rolling_buffer[-chunk.shape[0] :] = chunk
            samples_seen += chunk.shape[0]

            if samples_seen < runtime_config.stream_window_samples:
                continue

            predicted_index, confidence, probabilities = predict(
                model,
                rolling_buffer,
                runtime_config,
                feature_config,
                device,
            )
            if runtime_config.debug_top_k > 0:
                now_seconds = time.monotonic()
                if now_seconds - last_debug_emit >= runtime_config.debug_interval_seconds:
                    last_debug_emit = now_seconds
                    timestamp = datetime.now().isoformat(timespec="milliseconds")
                    print(
                        f"{timestamp} | DEBUG   | "
                        f"{debug_prediction_line(class_names, probabilities, runtime_config.debug_top_k)}"
                    )
            handle_prediction(
                class_names,
                predicted_index,
                confidence,
                runtime_config.threshold,
                runtime_config.class_thresholds,
                runtime_config.expected_class,
                last_emit_by_class,
                runtime_config.cooldown_seconds,
            )
    except KeyboardInterrupt:
        LOGGER.info("Stopping realtime inference.")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()


def main() -> int:
    configure_logging()
    try:
        runtime_config = parse_args()
        pyaudio_module = import_pyaudio()
        if runtime_config.list_devices:
            list_input_devices(pyaudio_module)
            return 0

        device = resolve_device()
        model, class_names, feature_config = load_model(runtime_config.model_path, device)
        run_stream(
            runtime_config,
            model,
            class_names,
            feature_config,
            device,
            pyaudio_module,
        )
    except Exception:
        LOGGER.exception("Realtime inference failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
