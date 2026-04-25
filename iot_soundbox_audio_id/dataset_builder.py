#!/usr/bin/env python3
"""Build the IoT Soundbox audio dataset from raw WAV recordings.

The builder implements the Phase 1 blueprint:
  * Read class folders from data/raw.
  * Slice audio into 1.0 second windows with 50% overlap.
  * Zero-pad short clips, including 0.5 second beeps, to 1.0 second.
  * Generate a mirrored 09_distortion sample for every clean functional
    sample from classes 01 through 06.

Output layout:
  data/processed/
      01_power_on/*.wav
      ...
      09_distortion/*.wav
      manifest.csv
      class_map.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import librosa
import numpy as np
import soundfile as sf


LOGGER = logging.getLogger("dataset_builder")

EXPECTED_CLASSES: tuple[str, ...] = (
    "01_power_on",
    "02_power_off",
    "03_beep",
    "04_otp",
    "05_charged",
    "06_bind",
    "07_silence",
    "08_interference",
    "09_distortion",
)
FUNCTIONAL_CLASSES: frozenset[str] = frozenset(EXPECTED_CLASSES[:6])
DISTORTION_CLASS = "09_distortion"


@dataclass(frozen=True)
class BuildConfig:
    """Runtime configuration for dataset generation."""

    raw_dir: Path
    output_dir: Path
    sample_rate: int = 44_100
    window_seconds: float = 1.0
    hop_seconds: float = 0.5
    random_seed: int = 17
    clean_output: bool = False
    distortion_min_snr_db: float = 6.0
    distortion_max_snr_db: float = 14.0
    distortion_min_clip_ratio: float = 0.22
    distortion_max_clip_ratio: float = 0.48

    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.sample_rate))

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_seconds * self.sample_rate))


@dataclass(frozen=True)
class ManifestRow:
    """Single generated training example entry."""

    path: str
    label_id: int
    class_name: str
    source_path: str
    source_class: str
    source_id: str
    start_seconds: float
    duration_seconds: float
    sample_rate: int
    is_synthetic: bool

    def as_csv_row(self) -> dict[str, str | int | float]:
        return {
            "path": self.path,
            "label_id": self.label_id,
            "class_name": self.class_name,
            "source_path": self.source_path,
            "source_class": self.source_class,
            "source_id": self.source_id,
            "start_seconds": f"{self.start_seconds:.6f}",
            "duration_seconds": f"{self.duration_seconds:.6f}",
            "sample_rate": self.sample_rate,
            "is_synthetic": int(self.is_synthetic),
        }


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Build 1-second IoT Soundbox audio windows and synthetic distortion mirrors."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing blueprint class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Destination directory for processed WAV windows and manifests.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44_100,
        help="Target sample rate. 44100 is recommended for realtime Mac EOL inference.",
    )
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output directory before writing the rebuilt dataset.",
    )
    args = parser.parse_args()

    return BuildConfig(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        random_seed=args.random_seed,
        clean_output=args.clean_output,
    )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_config(config: BuildConfig) -> None:
    if not config.raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {config.raw_dir}")
    if not config.raw_dir.is_dir():
        raise NotADirectoryError(f"Raw dataset path is not a directory: {config.raw_dir}")
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if config.window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if config.hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")
    if config.hop_samples > config.window_samples:
        raise ValueError("hop_seconds must be less than or equal to window_seconds")
    if not 0 < config.distortion_min_clip_ratio <= config.distortion_max_clip_ratio <= 1:
        raise ValueError("distortion clip ratios must be within (0, 1]")
    if config.distortion_min_snr_db > config.distortion_max_snr_db:
        raise ValueError("distortion_min_snr_db cannot exceed distortion_max_snr_db")


def prepare_output_dirs(config: BuildConfig) -> None:
    if config.clean_output and config.output_dir.exists():
        LOGGER.info("Removing existing output directory: %s", config.output_dir)
        shutil.rmtree(config.output_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / ".gitkeep").write_text("placeholder\n", encoding="utf-8")
    for class_name in EXPECTED_CLASSES:
        (config.output_dir / class_name).mkdir(parents=True, exist_ok=True)


def discover_wav_files(raw_dir: Path, class_name: str) -> list[Path]:
    class_dir = raw_dir / class_name
    if not class_dir.exists():
        LOGGER.warning("Missing raw class folder: %s", class_dir)
        return []
    if not class_dir.is_dir():
        LOGGER.warning("Expected class path is not a directory, skipping: %s", class_dir)
        return []
    return sorted(path for path in class_dir.rglob("*.wav") if path.is_file())


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load a WAV file as mono float32 audio at the target sample rate."""

    try:
        audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    except Exception as exc:  # noqa: BLE001 - preserve source filename in the raised error.
        raise RuntimeError(f"Failed to load audio file {path}: {exc}") from exc

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio file contains NaN or Inf values: {path}")

    return audio.astype(np.float32, copy=False)


def iter_windows(audio: np.ndarray, config: BuildConfig) -> Iterator[tuple[np.ndarray, int]]:
    """Yield fixed-length windows and their start sample.

    Complete windows are produced with a 0.5 second hop. Audio shorter than the
    1.0 second target is right-padded with zeros, which is required for 0.5s beeps.
    """

    window_samples = config.window_samples
    hop_samples = config.hop_samples

    if audio.shape[0] <= window_samples:
        yield zero_pad_or_trim(audio, window_samples), 0
        return

    last_start = audio.shape[0] - window_samples
    for start in range(0, last_start + 1, hop_samples):
        yield audio[start : start + window_samples].astype(np.float32, copy=False), start


def zero_pad_or_trim(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if audio.shape[0] == target_samples:
        return audio.astype(np.float32, copy=False)
    if audio.shape[0] > target_samples:
        return audio[:target_samples].astype(np.float32, copy=False)

    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: audio.shape[0]] = audio.astype(np.float32, copy=False)
    return padded


def synthesize_distortion(
    clean_audio: np.ndarray,
    rng: np.random.Generator,
    config: BuildConfig,
) -> np.ndarray:
    """Create a Class 09 mirror sample using hard clipping plus white noise."""

    peak = float(np.max(np.abs(clean_audio)))
    if peak <= 1e-8:
        base = clean_audio.copy()
        reference_rms = 1e-3
    else:
        clip_ratio = float(
            rng.uniform(config.distortion_min_clip_ratio, config.distortion_max_clip_ratio)
        )
        clip_threshold = max(peak * clip_ratio, 1e-4)
        base = np.clip(clean_audio, -clip_threshold, clip_threshold)
        reference_rms = max(root_mean_square(base), 1e-4)

    snr_db = float(rng.uniform(config.distortion_min_snr_db, config.distortion_max_snr_db))
    noise_rms = reference_rms / (10.0 ** (snr_db / 20.0))
    noise = rng.normal(loc=0.0, scale=noise_rms, size=base.shape).astype(np.float32)
    distorted = base.astype(np.float32, copy=False) + noise
    return np.clip(distorted, -1.0, 1.0).astype(np.float32, copy=False)


def root_mean_square(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def source_identifier(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def output_filename(
    source_path: Path,
    class_name: str,
    window_index: int,
    start_sample: int,
    sample_rate: int,
) -> str:
    safe_stem = sanitize_filename(source_path.stem)
    source_id = source_identifier(source_path)
    start_ms = int(round(start_sample / sample_rate * 1000.0))
    return f"{class_name}__{safe_stem}__{source_id}__w{window_index:04d}__{start_ms:06d}ms.wav"


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned[:80] if cleaned else "audio"


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(path, audio, sample_rate, subtype="PCM_16")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to write WAV file {path}: {exc}") from exc


def build_dataset(config: BuildConfig) -> list[ManifestRow]:
    validate_config(config)
    prepare_output_dirs(config)

    rng = np.random.default_rng(config.random_seed)
    class_to_id = {class_name: idx for idx, class_name in enumerate(EXPECTED_CLASSES)}
    rows: list[ManifestRow] = []

    raw_distortion_files = discover_wav_files(config.raw_dir, DISTORTION_CLASS)
    if raw_distortion_files:
        LOGGER.warning(
            "Ignoring %d raw %s files; blueprint Class 09 is generated synthetically.",
            len(raw_distortion_files),
            DISTORTION_CLASS,
        )

    for class_name in EXPECTED_CLASSES[:-1]:
        wav_files = discover_wav_files(config.raw_dir, class_name)
        LOGGER.info("Processing %s: %d WAV files", class_name, len(wav_files))
        for source_path in wav_files:
            rows.extend(process_source_file(source_path, class_name, class_to_id, rng, config))

    write_manifest(config.output_dir / "manifest.csv", rows)
    write_class_map(config.output_dir / "class_map.json", class_to_id, config)
    log_summary(rows)
    return rows


def process_source_file(
    source_path: Path,
    class_name: str,
    class_to_id: dict[str, int],
    rng: np.random.Generator,
    config: BuildConfig,
) -> list[ManifestRow]:
    audio = load_audio(source_path, config.sample_rate)
    source_id = source_identifier(source_path)
    rows: list[ManifestRow] = []

    for window_index, (window, start_sample) in enumerate(iter_windows(audio, config)):
        clean_filename = output_filename(
            source_path,
            class_name,
            window_index,
            start_sample,
            config.sample_rate,
        )
        clean_path = config.output_dir / class_name / clean_filename
        write_wav(clean_path, window, config.sample_rate)

        rows.append(
            make_manifest_row(
                output_path=clean_path,
                class_name=class_name,
                label_id=class_to_id[class_name],
                source_path=source_path,
                source_class=class_name,
                source_id=source_id,
                start_sample=start_sample,
                config=config,
                is_synthetic=False,
            )
        )

        if class_name in FUNCTIONAL_CLASSES:
            distortion = synthesize_distortion(window, rng, config)
            distortion_filename = clean_filename.replace(class_name, DISTORTION_CLASS, 1)
            distortion_path = config.output_dir / DISTORTION_CLASS / distortion_filename
            write_wav(distortion_path, distortion, config.sample_rate)
            rows.append(
                make_manifest_row(
                    output_path=distortion_path,
                    class_name=DISTORTION_CLASS,
                    label_id=class_to_id[DISTORTION_CLASS],
                    source_path=source_path,
                    source_class=class_name,
                    source_id=source_id,
                    start_sample=start_sample,
                    config=config,
                    is_synthetic=True,
                )
            )

    return rows


def make_manifest_row(
    output_path: Path,
    class_name: str,
    label_id: int,
    source_path: Path,
    source_class: str,
    source_id: str,
    start_sample: int,
    config: BuildConfig,
    is_synthetic: bool,
) -> ManifestRow:
    return ManifestRow(
        path=str(output_path.resolve()),
        label_id=label_id,
        class_name=class_name,
        source_path=str(source_path.resolve()),
        source_class=source_class,
        source_id=source_id,
        start_seconds=start_sample / config.sample_rate,
        duration_seconds=config.window_seconds,
        sample_rate=config.sample_rate,
        is_synthetic=is_synthetic,
    )


def write_manifest(path: Path, rows: Sequence[ManifestRow]) -> None:
    fieldnames = [
        "path",
        "label_id",
        "class_name",
        "source_path",
        "source_class",
        "source_id",
        "start_seconds",
        "duration_seconds",
        "sample_rate",
        "is_synthetic",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())
    LOGGER.info("Wrote manifest: %s", path)


def write_class_map(path: Path, class_to_id: dict[str, int], config: BuildConfig) -> None:
    payload = {
        "classes": [
            {"id": class_to_id[class_name], "name": class_name}
            for class_name in EXPECTED_CLASSES
        ],
        "sample_rate": config.sample_rate,
        "window_seconds": config.window_seconds,
        "hop_seconds": config.hop_seconds,
        "distortion_source": "synthetic_hard_clipping_plus_white_noise",
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    LOGGER.info("Wrote class map: %s", path)


def log_summary(rows: Iterable[ManifestRow]) -> None:
    counts: dict[str, int] = {class_name: 0 for class_name in EXPECTED_CLASSES}
    synthetic_count = 0
    total = 0
    for row in rows:
        counts[row.class_name] = counts.get(row.class_name, 0) + 1
        synthetic_count += int(row.is_synthetic)
        total += 1

    LOGGER.info("Generated %d total samples (%d synthetic)", total, synthetic_count)
    for class_name in EXPECTED_CLASSES:
        LOGGER.info("  %-16s %6d", class_name, counts.get(class_name, 0))


def main() -> int:
    configure_logging()
    config = parse_args()
    try:
        build_dataset(config)
    except Exception:
        LOGGER.exception("Dataset build failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
