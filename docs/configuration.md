# Configuration reference

Every command reads one TOML file (`-c`, default `configs/hey_joy.toml`). Relative paths are resolved from the directory you run the command in. Unknown keys in the settings sections are rejected, so typos fail loudly.

## `[wakeword]`

| Key | Default | Meaning |
| --- | --- | --- |
| `phrase` | required | Display name of the wake phrase. |
| `model_name` | slug of `phrase` | File name and score key of the model. `a-z`, `0-9` and `_` only. |
| `output_dir` | `runs/<model_name>` | Where clips, enrollment and models are written. |

## `[[positive]]` and `[[negative]]`

Repeatable tables, at least one of each.

| Key | Default | Meaning |
| --- | --- | --- |
| `text` | required | What Piper says. Punctuation changes intonation: "Hey Joy." and "Hey, Joy!" sound different. |
| `clips` | 10 | How many clips to synthesize. |
| `weight` | 1 | Negatives only: extra augmented copies of each clip during training. Use 3 or 4 for the closest confusables. |

## `[synthesis]`

| Key | Default | Meaning |
| --- | --- | --- |
| `voices` | lessac, amy, ryan | Piper voice names. Each needs `<voices_dir>/<name>.onnx` and its `.onnx.json`. |
| `voices_dir` | `voices` | Download with `python -m piper.download_voices --download-dir voices <name>`. |
| `length_scales` | `[0.85, 1.0, 1.15]` | Speaking-rate variations. |
| `noise_scales` | `[0.5, 0.667, 0.8]` | Intonation variations. |
| `noise_w_scales` | `[0.6, 0.8, 1.0]` | Phoneme-timing variations. |
| `max_speakers_per_voice` | 0 | Cap speakers used from multi-speaker voices (0 = all). |
| `seed` | 1234 | Changes which combinations each phrase gets. |

Voices missing from `voices_dir` are skipped with a download hint, and the remaining voices absorb their share of clips.

## `[background]`

| Key | Default | Meaning |
| --- | --- | --- |
| `dirs` | `[]` | Folders searched recursively for `.wav` files of non-wake audio. |
| `max_clips` | 2000 | Cap on 2-second slices taken across all files. |
| `hop_seconds` | 1.0 | Step between slices. |

## `[training]`

| Key | Default | Meaning |
| --- | --- | --- |
| `epochs` | 30 | Maximum epochs. |
| `hidden_dim` | 64 | Width of both hidden layers. |
| `learning_rate` | 0.0008 | Adam learning rate. |
| `batch_size` | 64 | Minibatch size. |
| `negative_weight` | 2.2 | Loss multiplier on negative windows. Raise it to trade recall for fewer false triggers. |
| `positive_copies` | 2 | Augmented copies of each positive clip. |
| `noise_clips` | 12 | Pure-noise negatives added to training. |
| `holdout_fraction` | 0.15 | Share of clips held out for validation and `evaluate`. |
| `patience` | 8 | Early-stopping patience in epochs (needs a holdout). |
| `seed` | 7 | Seed for augmentation and weight init. |

## `[listen]`

| Key | Default | Meaning |
| --- | --- | --- |
| `threshold` | 0.65 | Score that counts as a detection. Override per run with `--threshold`. |
| `vad_threshold` | 0.35 | Silero VAD gate used while listening (0 disables it). |
| `refractory_seconds` | 2.2 | Quiet period after a detection; also used for false accepts/hour. |
| `ack_text` | `"Yes?"` | What Piper says back when the wake word fires. |

## Using the model elsewhere

The exported `<model_name>.onnx` is a standard openWakeWord model:

```python
from openwakeword.model import Model

model = Model(wakeword_models=["runs/hey_joy/models/hey_joy.onnx"], inference_framework="onnx")
scores = model.predict(frame_int16_1280_samples)  # {"hey_joy": 0.97}
```
