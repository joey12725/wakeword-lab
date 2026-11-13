"""Evaluate a trained model the way it will actually run: streaming.

Two checks:

* **Holdout clips.** Every clip the trainer never saw is streamed through
  ``openwakeword.Model`` frame by frame. The peak score per clip decides
  whether it fired. Results are grouped by phrase, so you can see exactly
  which confusable ("Hey Joey", "Enjoy") is the weak spot.
* **False accepts per hour.** Optionally, run the classifier over a long
  stream of precomputed openWakeWord features of ordinary speech, music and
  noise (for example openWakeWord's ``validation_set_features.npy``) and
  count distinct activations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import FRAME_SAMPLES, N_FRAMES, load_wav_16k, to_int16
from .config import WakeWordConfig
from .dataset import is_holdout
from .features import WARMUP_FRAMES, ensure_resources
from .metrics import (
    FRAME_SECONDS,
    count_activations,
    false_accepts_per_hour,
    format_table,
    overall,
    sliding_windows,
    summarize_peaks,
)
from .synth import read_manifest


def clip_peaks(config: WakeWordConfig, include_all: bool = False, log=print) -> list[tuple[str, int, float]]:
    from openwakeword.model import Model

    ensure_resources()
    model = Model(wakeword_models=[str(config.model_path)], inference_framework="onnx", vad_threshold=0.0)
    silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)

    clips = read_manifest(config.clips_dir)
    if not include_all:
        clips = [c for c in clips if is_holdout(c.path, config.training.holdout_fraction)]
    if not clips:
        raise SystemExit("No clips to evaluate. Run `wakeword-lab synth` first, or pass --all.")

    rows = []
    for clip in clips:
        model.reset()
        for _ in range(WARMUP_FRAMES):
            model.predict(silence)
        preds = model.predict_clip(to_int16(load_wav_16k(config.clips_dir / clip.path)), padding=0)
        peak = max((float(p[config.model_name]) for p in preds), default=0.0)
        rows.append((clip.text, clip.label, peak))
    log(f"  scored {len(rows)} {'' if include_all else 'holdout '}clips")
    return rows


def stream_false_accepts(
    model_path: Path,
    features_path: Path,
    threshold: float,
    refractory_seconds: float,
    batch: int = 4096,
) -> dict[str, float]:
    import onnxruntime as ort

    feats = np.load(features_path, mmap_mode="r")
    windows = sliding_windows(feats, N_FRAMES)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    name = inp.name
    if inp.shape[0] == 1:  # models exported with a fixed batch of 1 (e.g. openWakeWord's stock models)
        batch = 1
    scores = np.concatenate(
        [
            session.run(None, {name: np.ascontiguousarray(windows[i : i + batch], dtype=np.float32)})[0].reshape(-1)
            for i in range(0, len(windows), batch)
        ]
    )
    refractory = int(round(refractory_seconds / FRAME_SECONDS))
    n_fa = count_activations(scores, threshold, refractory)
    return {
        "hours": len(scores) * FRAME_SECONDS / 3600.0,
        "false_accepts": n_fa,
        "false_accepts_per_hour": false_accepts_per_hour(n_fa, len(scores)),
    }


def evaluate(
    config: WakeWordConfig,
    threshold: float | None = None,
    include_all: bool = False,
    fp_features: Path | None = None,
    log=print,
) -> dict:
    if not config.model_path.exists():
        raise SystemExit(f"No model at {config.model_path}. Run `wakeword-lab train` first.")
    threshold = config.listen.threshold if threshold is None else threshold

    results = summarize_peaks(clip_peaks(config, include_all, log), threshold)
    summary = overall(results)
    log("")
    log(format_table(results, threshold))
    log("")
    log(f"  recall {summary['recall']:.1%} on {summary['positive_clips']} wake-phrase clips")
    log(f"  false accepts {summary['false_accept_rate']:.1%} on {summary['negative_clips']} confusable clips")

    report: dict = {"threshold": threshold, "clips": summary, "phrases": [r.__dict__ for r in results]}
    if fp_features:
        fa = stream_false_accepts(config.model_path, fp_features, threshold, config.listen.refractory_seconds)
        report["stream"] = fa
        log(f"  {fa['false_accepts_per_hour']:.2f} false accepts/hour over {fa['hours']:.1f} h of background audio")

    out = config.model_path.with_suffix(".eval.json")
    out.write_text(json.dumps(report, indent=2))
    log(f"  wrote {out}")
    return report
