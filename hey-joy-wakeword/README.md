# hey-joy-wakeword

Custom OpenWakeWord model trained to respond to "hey Joy".

## Training Details
- **Target phrase:** "hey Joy"
- **Adversarial negatives:** "hey Joey", "hey Joe", "hey Joyce", "enjoy", "ahoy", "joy", "Joey"
- **Positive data:** ~5000 synthetic TTS clips (VITS + Waveglow, multiple voices)
- **Negative data:** ~30,000 hours (Common Voice speech, FMA music, FSD50k noise)
- **Model:** Fully-connected neural network → ONNX/tflite
- **Hardware:** NVIDIA RTX 3080 (CUDA)
- **Training time budget:** ≤16 hours

## Usage
```python
import openwakeword
from openwakeword.model import Model

model = Model(wakeword_models=["hey_joy.onnx"])
prediction = model.predict(audio_frame)
```

## Structure
- `train.py` — Full training pipeline (data gen → training → export)
- `generate_positive.py` — Synthetic TTS data generation for "hey Joy"
- `generate_adversarial.py` — Adversarial negative generation ("hey Joey" etc.)
- `models/` — Trained model outputs
- `data/` — Training data (not committed)
