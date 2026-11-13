# How the pipeline works

wakeword-lab is four stages glued together by one TOML file. Each stage writes into `runs/<model_name>/` and the next one reads from there, so you can rerun any stage on its own.

```
runs/hey_joy/
  clips/
    manifest.csv            one row per planned clip
    positive/00_hey_joy/…   synthesized wake-phrase clips
    negative/00_hey_joey/…  synthesized confusables
    ack.wav                 the "Yes?" played on detection
  enroll/                   optional takes of your own voice (local only)
  models/
    hey_joy.onnx            the classifier openWakeWord loads
    hey_joy.train.json      per-epoch loss / holdout metrics
    hey_joy.eval.json       per-phrase scorecard
```

## 1. Synthetic data (`synth.py`)

A wake-word model needs two things a person can't easily record: thousands of different voices saying the phrase, and thousands of near-misses. Piper produces both.

**Planning.** For each `[[positive]]` and `[[negative]]` entry the planner builds the full grid of

```
(voice, speaker) × length_scale × noise_scale × noise_w_scale
```

shuffles it with a seed derived from the phrase, and walks it in order. Small clip counts therefore still touch every voice, and large counts only repeat a combination after the grid is exhausted. Multi-speaker voices contribute one grid row per speaker, so `en_US-libritts_r-medium` alone gives 904 distinct speakers.

- `length_scale` changes speaking rate (0.8 is brisk, 1.25 is slow and deliberate).
- `noise_scale` and `noise_w_scale` change intonation and phoneme timing.

**Resumability.** The plan is written to `manifest.csv` before synthesis starts, and files already on disk are skipped. The manifest also records each clip's label and weight, which training reads back later.

**Background audio.** `wakeword-lab negatives` downloads openWakeWord's published samples of FMA (music) and FSD50K (sound effects). Any folder of 16 kHz WAV files works; list them under `[background] dirs` and they are sliced into 2-second windows.

## 2. Features and training (`features.py`, `dataset.py`, `train.py`)

**Embeddings, not audio.** openWakeWord's front end turns every 80 ms frame into a 96-dimensional speech embedding (melspectrogram → a pretrained embedding network trained on a very large speech corpus). A custom wake word is a small classifier over the most recent 16 embeddings, about 1.3 seconds of context. Because the expensive part is pretrained, the head trains in minutes and generalizes far beyond the voices it was trained on.

**Streaming extraction.** Each clip is fed frame by frame through `AudioFeatures`, after 50 frames of warm-up silence, exactly as the live `Model` does. Every frame yields one `(16, 96)` window.

**Picking the right windows.** A 2-second positive clip is mostly silence. Labelling every window "wake word" would teach the model that silence is the wake word. Only windows that end on a frame within 22 % of the clip's peak energy are kept as positives. Negatives keep every window.

**Holdout split.** Each clip goes to train or holdout by a SHA-1 of its relative path. The split is decided before augmentation, so no augmented copy of a holdout clip ever reaches training, and it stays the same across reruns.

**Augmentation.** Each training clip gets `weight` extra copies (negatives) or `positive_copies` copies (positives) with random gain (0.55 to 1.35x), Gaussian hiss and up to ±150 ms time shift. A dozen pure-noise clips and one silent clip teach the model that a quiet room is not a wake word.

**Model.**

```
Flatten(16×96) → Linear(1536, 64) → LayerNorm → ReLU
               → Linear(64, 64)   → LayerNorm → ReLU
               → Linear(64, 1)    → Sigmoid
```

About 103k parameters. The loss is binary cross-entropy with negatives weighted `negative_weight` (2.2 by default): a wake word that fires during a conversation is far more annoying than one that occasionally needs a second try. The best epoch by holdout loss is kept (early stopping with `patience`), then exported to ONNX with the output named after the model so openWakeWord reports scores under that key.

## 3. Evaluation (`evaluate.py`, `metrics.py`)

**Per-phrase scorecard.** Every holdout clip is streamed through `openwakeword.Model` (VAD off) and its peak score recorded. Rows are grouped by phrase:

- for the wake phrase, the share of clips that crossed the threshold is **recall**;
- for each negative, that share is its **false-accept rate**, and the table sorts the worst offender to the top.

That ordering makes the next step obvious: if "Hey Joey" is at 8 %, add more "Hey Joey" clips or raise its weight.

**False accepts per hour.** Clip-level rates say nothing about a TV playing all evening. `--fp-features` takes a long `(T, 96)` array of precomputed embeddings (openWakeWord's `validation_set_features.npy` is about 11 hours of speech, music and noise), slides 16-frame windows across it, scores them in batches with ONNX Runtime, and counts distinct activations using the live listener's refractory period.

## 4. Live detection (`devices.py`, `listen.py`)

- **Capture.** The device is opened at 16 kHz mono if it supports it, otherwise at 48 / 44.1 kHz or its native rate, with as many channels as needed to reach the requested one.
- **Resampling.** A streaming soxr resampler keeps filter state between blocks, and `FrameChunker` emits exact 1280-sample frames regardless of the device's block size.
- **Scoring.** Frames go to `openwakeword.Model` with Silero VAD gating (`vad_threshold`), which suppresses scores during non-speech.
- **Trigger gate.** A detection fires when the score crosses the threshold. The gate then ignores everything for `refractory_seconds` *and* until the score falls below 40 % of the threshold, so one long "Heeey Joy" never fires twice.
- **Mic probe.** `--probe` records a second from every plausible input (skipping virtual and loopback devices) and picks the loudest channel. On a multi-input audio interface this finds the right mic without guessing indices.

## Enrollment

`--enroll` records five takes of the phrase and six seconds of room tone through the chosen mic. Takes are weighted heavily (10 augmented copies each) and the room tone is sliced into negatives, which tunes the model to your voice, your accent and your mic's coloration. The files live in `runs/<name>/enroll/` and are reused on later trainings until you delete them. Keep them out of version control; `.gitignore` already does.
