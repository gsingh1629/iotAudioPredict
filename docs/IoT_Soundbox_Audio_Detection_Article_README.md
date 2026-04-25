# From Manual Listening to Edge Audio QA: Building an Audio Detection System for IoT Soundbox Testing

**A practical journey into audio ML for real-world hardware quality checks**

By Gurpreet Singh

Co-author: Codex 😊

Repository: https://github.com/gsingh1629/iotAudioPredict

## The Problem I Wanted to Solve

During mass production of IoT devices and other hardware products, factories still depend on human effort for many acoustic checks. A device plays a sound, an operator listens, and the unit is marked as passed or failed. This works when volume is low, but it becomes fragile when hundreds or thousands of devices need to be tested repeatedly.

My immediate use case was an IoT payment soundbox. These devices play important audio events such as power on, power off, OTP prompts, beeps, charging messages, and binding notifications. If a device plays the wrong sound, produces a distorted sound, or fails to play audio clearly, that should be caught before shipping.

Manual listening creates a possibility of error. People get tired. Factory environments are noisy. Speaker damage can be subtle. Similar-sounding prompts can be confused. A unit may technically make sound but still have clipping, crackle, or an incorrect prompt. I wanted to reduce this manual dependency by building a repeatable audio detection pipeline.

> Goal: build a local system that listens to a hardware device during End-of-Line testing and automatically identifies whether the expected sound was played.

## Why Audio Detection Is Useful in Hardware Quality Testing

Audio is part of the user experience for many hardware products: payment devices, appliances, medical devices, industrial alert modules, smart speakers, automotive accessories, and security systems. If sound is part of the product behavior, it should also be part of the production test plan.

A good acoustic test should answer several questions. Did the speaker work? Was the correct prompt played? Was the sound distorted? Was the volume acceptable? Was the device silent when it should not have been? Did background noise or interference trigger a false result?

This is where machine learning becomes useful. Rather than testing only amplitude or duration, an audio classifier can learn the pattern of each sound and separate valid product audio from silence, interference, and hardware-level distortion.

## How Audio Classification Works at a High Level

Raw audio is a waveform: amplitude changing over time. Neural networks can learn from raw waveforms, but a very common practical approach is to convert audio into a time-frequency representation called a spectrogram.

For this project I used Mel-spectrograms. A Mel-spectrogram represents how frequency energy changes over time using a scale that is closer to human hearing. Once audio is converted into a Mel-spectrogram, the classification problem becomes similar to image classification.

That means a Convolutional Neural Network can learn patterns such as: this frequency-time shape looks like a power-on prompt, this one looks like a beep, this one looks like speaker distortion, and this one is just factory interference.

```text
Audio window → Mel-spectrogram → CNN → Class prediction → EVENT / FAILURE
```

## My Use Case: IoT Soundbox End-of-Line Testing

The current prototype focuses on identifying the sounds of an IoT payment soundbox. The model runs locally on my MacBook and listens through the MacBook microphone. The soundbox plays an audio event, and the inference engine predicts the class in real time.

The classes are intentionally designed for production testing. The first six classes represent valid functional sounds. The remaining classes represent conditions where the system should not emit a normal success event.

| Class | Meaning | Why it matters |
| --- | --- | --- |
| 01_power_on | Device startup sound | Confirms startup audio path and correct prompt |
| 02_power_off | Device shutdown sound | Confirms shutdown prompt is present and recognizable |
| 03_beep | Short beep interaction | Covers brief feedback sounds that need padding/windowing |
| 04_otp | OTP prompt | Checks prompt classification for payment flows |
| 05_charged | Battery/charged prompt | Covers charging related audio feedback |
| 06_bind | Pairing/bind prompt | Checks sync or binding notification audio |
| 07_silence | No useful product sound | Prevents false firing when the environment is quiet |
| 08_interference | Non-product noise | Rejects talking, handling noise, tools, and factory background |
| 09_distortion | Synthetic distorted product sound | Flags likely speaker or audio-path failure |

## Data Collection: What I Recorded and Why

The first version of the dataset used WAV recordings from the actual soundbox. I recorded functional sounds such as power on, power off, beep, OTP, battery prompts, and bind prompts. I also recorded silence and interference samples because a real inference system must learn what not to detect.

Some files were recorded using a laptop microphone and some using a mobile microphone. This does not mean the audio was played from a phone or laptop speaker. The sound still came from the soundbox. Mobile and laptop were only recording devices used to introduce microphone variation.

For production, I would make the data collection process more systematic: multiple hardware units, multiple volume levels, different jig positions, repeated takes, and more factory noise conditions. The current dataset was enough for a working prototype, but the long-term quality of the system will depend heavily on real-world data diversity.

> Important rule: the folder is the label. The filename stores metadata such as volume, recorder, device unit, position, and take number.

## Dataset Folder Structure

I kept the raw dataset structure simple and machine-readable. Each class has its own folder under data/raw. This makes the data builder deterministic and avoids mixing labels with recording metadata.

```text
data/raw/
  01_power_on/
  02_power_off/
  03_beep/
  04_otp/
  05_charged/
  06_bind/
  07_silence/
  08_interference/
  09_distortion/
```

The 09_distortion folder is intentionally kept empty in raw data. Distortion samples are generated synthetically from clean functional samples. This lets the model learn a mirror class: a distorted power-on or beep should not be treated as a normal pass.

## Dataset Creation Pipeline

Raw audio files are not used directly. I built a dataset builder that turns raw WAV files into fixed one-second training examples. This is important because neural networks need consistent input shapes and real-time inference also works on a rolling one-second buffer.

The pipeline reads WAV files, converts them to mono, resamples them to 44.1 kHz, slices longer files using a sliding window, pads short beeps, and writes processed examples with a manifest and class map.

| Step | Implementation | Purpose |
| --- | --- | --- |
| Read raw WAV files | Parse data/raw/<class_name> recursively | Keeps the folder as the class label |
| Normalize shape | Mono audio, target sample rate | Makes training and inference consistent |
| Sliding windows | 1.0 second window, 0.5 second overlap | Creates multiple examples from longer prompts |
| Beep handling | Pad short clips to 1.0 second | Preserves short sounds without changing model input size |
| Distortion mirror | Hard clipping plus white noise | Teaches failure detection for damaged or noisy audio |
| Manifest output | CSV plus class map JSON | Makes training reproducible |

## Model Training Setup

The model is a compact Audio CNN trained on Mel-spectrogram inputs. The training loop is written in PyTorch and explicitly uses Apple Silicon MPS acceleration on my MacBook.

Training includes dynamic waveform augmentations: small pitch shifts, slight time stretching, and gain randomization. These augmentations help the model become less brittle when the same product sound is captured at different volumes or through slightly different microphones.

The training script saves the best validation checkpoint, not just the final epoch. This is important because small audio datasets can overfit quickly and validation performance may fluctuate.

```text
uv run --no-editable soundbox-train --epochs 50 --batch-size 16
```

## V1: When the First Model Was Not Enough

The first model worked partially. It detected power-on and beep reliably, but power-off was inconsistent during live inference. At first, this looked like a model failure.

The useful breakthrough was adding a debug mode that prints the top class probabilities. That showed power-off was appearing in the top predictions, but with low confidence. The model was directionally correct but not confident enough to pass the event threshold.

The first useful checkpoint had validation accuracy around 0.80. Offline, the model correctly recognized processed power-off windows, but the average confidence for power-off was only around 0.61. This explained why live inference missed it at stricter thresholds.

> Lesson: accuracy alone is not enough for real-time systems. Confidence distribution matters.

## V2: Runtime Calibration and Expected-Class Testing

For the second version, I added class-specific thresholds. Instead of forcing every class to use the same confidence threshold, I allowed power-off and beep to have prototype-specific thresholds while keeping distortion strict.

I also added expected-class filtering. In an End-of-Line test, the orchestrator often knows what sound should play next. If the test step is power-off, the inference engine can emit only power-off events while still flagging distortion failures. This makes the model more practical inside a controlled production test sequence.

```text
uv run --no-editable soundbox-infer --expected-class 02_power_off --debug
```

## Training Longer for the Final Prototype Model

After V2, I trained a longer candidate model for 100 epochs with patience. The important detail is that the script saves the best validation checkpoint, so the final model is not simply whatever happened at epoch 100.

The earlier model reached about 0.80 validation accuracy. The longer training run produced a better checkpoint at epoch 79 with validation accuracy around 0.90. More importantly, the average top confidence for power-off improved from roughly 0.61 to roughly 0.936 on processed windows.

This made live power-off detection much more usable. It also reinforced that training longer can help, but only when combined with validation, best-checkpoint saving, and careful live debugging.

| Version | Best Epoch | Validation Accuracy | Observation |
| --- | --- | --- | --- |
| V1 | 17 | 0.80 | Detected power-on and beep, but power-off was under-confident |
| V2 long run | 79 | 0.90 | Power-off confidence improved significantly and live detection became usable |

## Real-Time Inference Pipeline

The real-time inference script uses PyAudio to continuously listen through the MacBook Pro microphone. It keeps a rolling one-second buffer and runs prediction every half second. The MacBook microphone captures at 48 kHz, and the pipeline resamples to the model’s 44.1 kHz feature configuration.

The current command is intentionally simple. The repository includes the current best model checkpoint, so someone can clone the repo, install dependencies, and immediately run inference.

```text
uv sync --no-editable
uv run --no-editable soundbox-infer
uv run --no-editable soundbox-infer --debug
```

## What the Repository Contains

I cleaned the project into a reusable Python package with short CLI commands. The repository includes the source code, UV lockfile, dataset folder placeholders, documentation, and the current default inference model. Raw audio and processed datasets are kept local and ignored by Git.

| Component | Purpose |
| --- | --- |
| soundbox-build | Creates processed one-second windows and synthetic distortion samples |
| soundbox-train | Trains the Mel-spectrogram CNN using PyTorch and MPS |
| soundbox-infer | Runs real-time microphone inference with prototype or strict profiles |
| models/iot_soundbox_audio_cnn.pth | Current best default model for immediate inference |
| data/raw/README.md | Explains how to place raw recordings for retraining |

## Future Direction: From Mac Prototype to Independent Edge Device

Right now the system runs on my MacBook. That is useful for prototyping, but the next goal is to make it a dedicated test device.

The future setup would use a separate microphone and an edge device running local inference. The testing jig would trigger the product sound, the microphone would capture audio, the edge device would classify it, and the result would be sent as PASS, FAIL, WRONG SOUND, or DISTORTION.

This would make the solution independent of a laptop and much more practical for factory deployment.

```text
Testing jig → Product plays sound → Microphone captures audio → Edge model inference → PASS / FAIL / WRONG SOUND / DISTORTION
```

## Making It Useful Beyond One Product

The larger goal is not just to build one model for one soundbox. The more interesting goal is to create a reusable sound detection tool for hardware products.

A team should be able to collect their own product-specific audio, place it into class folders, run the dataset builder, train a model, and deploy local inference. That could apply to payment devices, appliances, medical hardware, industrial alert systems, or any device where sound output is part of quality assurance.

To make this fully production-ready, the biggest future task is data. More devices, more environments, more real interference, more failure cases, and more real distorted speaker samples will make the system stronger and more reliable.

> The long-term vision: train a reliable audio QA model for any hardware product by collecting the right product-specific dataset.

## Final Thoughts

This project started from a very practical observation: factories still rely on human listening for some acoustic quality checks. That creates room for error, especially at scale.

By combining structured data collection, audio preprocessing, Mel-spectrograms, CNN training, class-specific calibration, and real-time inference, I built an end-to-end prototype that can classify soundbox audio events locally.

It is not a finished industrial product yet, but it is a strong foundation. The next step is to gather more production-like data and move inference to an independent edge device. This is the kind of real-world AI problem I enjoy: small enough to prototype, but connected directly to manufacturing quality, hardware reliability, and practical automation.
