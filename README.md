# IoT Soundbox Audio Identification

Production-oriented audio classification pipeline for IoT Soundbox EOL acoustic verification.

Author: Gurpreet Singh  
Co-author: Codex 😊

## Setup

Install system audio dependency on macOS:

```bash
brew install portaudio
```

Install Python dependencies with `uv`:

```bash
brew install uv
uv sync --no-editable
```

The repository includes `.python-version` so `uv` uses Python 3.14 for this
project environment.

## Data

Place raw WAV files under `data/raw/<class_name>/`. Folder names are labels:

```text
01_power_on
02_power_off
03_beep
04_otp
05_charged
06_bind
07_silence
08_interference
09_distortion
```

Keep `09_distortion` empty. It is generated automatically from clean classes `01` through `06`.

## Build Dataset

```bash
uv run --no-editable soundbox-build --clean-output
```

## Train

```bash
uv run --no-editable soundbox-train --epochs 50 --batch-size 16
```

The best validation checkpoint is saved to:

```text
models/iot_soundbox_audio_cnn.pth
```

The repository includes the current best checkpoint at that path so inference
can run immediately after setup. Re-training will overwrite it locally.

## Realtime Inference

List microphone devices:

```bash
uv run --no-editable soundbox-infer --list-devices
```

Run inference:

```bash
uv run --no-editable soundbox-infer
```

By default, realtime inference prefers the MacBook Pro microphone and captures at
48 kHz, then resamples each one-second window to the model's 44.1 kHz feature
pipeline.

Debug live predictions:

```bash
uv run --no-editable soundbox-infer --debug
```

Test only the power-off event:

```bash
uv run --no-editable soundbox-infer --expected-class 02_power_off --debug
```
