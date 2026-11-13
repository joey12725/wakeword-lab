"""Detection metrics. Pure NumPy so they are cheap to test and reuse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import FRAME_SAMPLES, SAMPLE_RATE

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


def count_activations(scores: np.ndarray, threshold: float, refractory_frames: int = 0) -> int:
    """Count distinct detections in a score stream.

    A detection starts on the first frame at or above ``threshold``; the next
    ``refractory_frames`` frames are ignored, matching how the live listener
    suppresses repeat triggers.
    """
    hits = 0
    i = 0
    above = np.flatnonzero(np.asarray(scores) >= threshold)
    for idx in above:
        if idx < i:
            continue
        hits += 1
        i = idx + 1 + refractory_frames
    return hits


def false_accepts_per_hour(n_false_accepts: int, n_frames: int, frame_seconds: float = FRAME_SECONDS) -> float:
    hours = n_frames * frame_seconds / 3600.0
    return n_false_accepts / hours if hours > 0 else 0.0


def sliding_windows(features: np.ndarray, n_frames: int) -> np.ndarray:
    """(T, D) per-frame embeddings -> (T - n + 1, n, D) model inputs.

    openWakeWord publishes long negative sets as one flat (T, 96) array; the
    classifier wants overlapping 16-frame windows.
    """
    features = np.asarray(features)
    if features.ndim == 3:
        return features
    if features.ndim != 2 or features.shape[0] < n_frames:
        raise ValueError(f"expected (T>={n_frames}, D) features, got {features.shape}")
    return np.lib.stride_tricks.sliding_window_view(features, n_frames, axis=0).transpose(0, 2, 1)


@dataclass(frozen=True)
class PhraseResult:
    text: str
    label: int  # 1 = wake phrase, 0 = should be rejected
    clips: int
    triggered: int
    mean_peak: float

    @property
    def trigger_rate(self) -> float:
        return self.triggered / self.clips if self.clips else 0.0


def summarize_peaks(rows: list[tuple[str, int, float]], threshold: float) -> list[PhraseResult]:
    """Group (text, label, peak score) rows by phrase.

    Positives are listed first, then negatives from most to least confusing.
    """
    groups: dict[tuple[str, int], list[float]] = {}
    for text, label, peak in rows:
        groups.setdefault((text, int(label)), []).append(float(peak))
    results = [
        PhraseResult(
            text=text,
            label=label,
            clips=len(peaks),
            triggered=sum(p >= threshold for p in peaks),
            mean_peak=float(np.mean(peaks)),
        )
        for (text, label), peaks in groups.items()
    ]
    return sorted(results, key=lambda r: (-r.label, -r.trigger_rate, -r.mean_peak, r.text))


def overall(results: list[PhraseResult]) -> dict[str, float]:
    pos = [r for r in results if r.label == 1]
    neg = [r for r in results if r.label == 0]
    pos_clips = sum(r.clips for r in pos)
    neg_clips = sum(r.clips for r in neg)
    return {
        "recall": sum(r.triggered for r in pos) / pos_clips if pos_clips else 0.0,
        "false_accept_rate": sum(r.triggered for r in neg) / neg_clips if neg_clips else 0.0,
        "positive_clips": pos_clips,
        "negative_clips": neg_clips,
    }


def format_table(results: list[PhraseResult], threshold: float) -> str:
    width = max([len("phrase")] + [len(r.text) for r in results])
    fired = f"fired @{threshold:.2f}"
    lines = [
        f"  {'phrase':<{width}}  {'want':>4}  {'clips':>5}  {'mean peak':>9}  {fired:>10}",
        "  " + "-" * (width + 38),
    ]
    for r in results:
        want = "fire" if r.label else "skip"
        lines.append(f"  {r.text:<{width}}  {want:>4}  {r.clips:>5}  {r.mean_peak:>9.3f}  {r.trigger_rate:>9.1%}")
    return "\n".join(lines)
