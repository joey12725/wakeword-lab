# Hey Joy Wake-Word Demo

An interactive Windows prototype for training and testing a small, local `openWakeWord`-compatible ONNX classifier for the phrase **“Hey Joy.”** It can synthesize positive and negative training clips with Piper, optionally enroll from the microphone, train a compact model, evaluate it, and listen for the wake phrase in real time.

## Repository contents

- `demo.py` — training, enrollment, evaluation, and live-listening workflow.
- `listen.bat` — convenience launcher for the known microphone setup.
- `run_demo.bat` — convenience launcher for interactive training/listening.

Generated clips, trained models, local Piper sources, downloaded voices, and Python environments are intentionally excluded. The working folder contained roughly 1.5 GB of generated/model/third-party data, so this repository keeps only the reproducible wrapper code.

## Setup

Use Python 3.10+ and install the audio/ML dependencies in the environment used by the demo. A CUDA-capable PyTorch install is recommended for training; inference uses ONNX on CPU.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Install Piper separately and place its voice files where the local configuration expects them, or update `PIPER_DIR`/`PIPER_VOICES` in `demo.py`. The `openWakeWord` package downloads its VAD assets on first use.

## Usage

```bash
python demo.py --list-devices
python demo.py --generate-tts
python demo.py --train-only
python demo.py --skip-enroll --device 3 --channel 0
```

The default workflow records a short microphone enrollment, trains `models/hey_joy.onnx`, evaluates it, and starts listening. Use `--retrain` to force a new model and `--threshold` to adjust detection sensitivity.

## Privacy and licensing

Microphone recordings and generated audio may contain personal data; keep them local. Review the licenses of Piper, openWakeWord, model weights, and any voice files before redistribution. This wrapper is a personal experiment with no separate license selected yet.
