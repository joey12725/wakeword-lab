"""Turn synthetic clips, enrollment takes and background audio into
(window, label) arrays for training.

Clips are split into train and holdout sets by a stable hash of their file
name, before any augmentation, so no holdout clip ever leaks into training
in augmented form.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import CLIP_SAMPLES, SAMPLE_RATE, load_wav_16k
from .config import WakeWordConfig
from .features import extract_windows
from .synth import ClipJob


def is_holdout(name: str, fraction: float) -> bool:
    """Deterministically assign ``fraction`` of names to the holdout set."""
    if fraction <= 0:
        return False
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def augment_clip(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random gain, background hiss and time shift."""
    out = np.asarray(audio, dtype=np.float32) * float(rng.uniform(0.55, 1.35))
    if rng.random() < 0.7:
        out = out + rng.normal(0, rng.uniform(0.004, 0.03), out.shape).astype(np.float32)
    if rng.random() < 0.4:
        out = np.roll(out, int(rng.integers(-2400, 2400)))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def slice_long_audio(audio: np.ndarray, hop: int, length: int = CLIP_SAMPLES) -> list[np.ndarray]:
    """Cut long recordings into fixed-length negative clips."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size <= length:
        return [audio]
    return [audio[s : s + length] for s in range(0, audio.size - length + 1, hop)]


def background_files(dirs: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        files += sorted(Path(d).rglob("*.wav"))
    return files


@dataclass
class WindowSet:
    X: list[np.ndarray] = field(default_factory=list)
    y: list[int] = field(default_factory=list)

    def add(self, windows: list[np.ndarray], label: int) -> None:
        self.X += windows
        self.y += [label] * len(windows)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.X:
            return np.zeros((0, 16, 96), np.float32), np.zeros(0, np.float32)
        return np.stack(self.X).astype(np.float32), np.asarray(self.y, dtype=np.float32)


def build_windows(
    config: WakeWordConfig,
    clips: list[ClipJob],
    extractor,
    enroll_pos: list[np.ndarray] = (),
    enroll_neg: list[np.ndarray] = (),
    log=print,
) -> tuple[WindowSet, WindowSet]:
    """Return (train, holdout) window sets."""
    tr = config.training
    rng = np.random.default_rng(tr.seed)
    train, holdout = WindowSet(), WindowSet()

    def add(audio: np.ndarray, label: int, copies: int, target: WindowSet) -> None:
        for variant in [audio] + [augment_clip(audio, rng) for _ in range(copies)]:
            target.add(extract_windows(variant, extractor, positive=label == 1), label)

    for i, clip in enumerate(clips, 1):
        audio = load_wav_16k(config.clips_dir / clip.path)
        if is_holdout(clip.path, tr.holdout_fraction):
            add(audio, clip.label, 0, holdout)
        else:
            copies = tr.positive_copies if clip.label else clip.weight
            add(audio, clip.label, copies, train)
        if i % 200 == 0 or i == len(clips):
            log(f"    embedded {i}/{len(clips)} clips")

    # Your own voice is the most valuable data there is: weight it heavily.
    for audio in enroll_pos:
        add(audio, 1, 10, train)
    for audio in enroll_neg:
        for piece in slice_long_audio(audio, SAMPLE_RATE // 2):
            add(piece, 0, 1, train)

    bg = config.background
    bg_hop = int(bg.hop_seconds * SAMPLE_RATE)
    n_bg = 0
    for path in background_files(bg.dirs):
        for piece in slice_long_audio(load_wav_16k(path), bg_hop):
            if n_bg >= bg.max_clips:
                break
            target = holdout if is_holdout(f"{path}:{n_bg}", tr.holdout_fraction) else train
            add(piece, 0, 0, target)
            n_bg += 1
        if n_bg >= bg.max_clips:
            break
    if n_bg:
        log(f"    added {n_bg} background clips")

    # Silence and hiss, so a quiet room never fires.
    for _ in range(tr.noise_clips):
        add(rng.normal(0, rng.uniform(0.002, 0.02), CLIP_SAMPLES).astype(np.float32), 0, 0, train)
    add(np.zeros(CLIP_SAMPLES, dtype=np.float32), 0, 0, train)

    return train, holdout
