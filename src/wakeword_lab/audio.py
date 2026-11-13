"""Audio helpers shared by synthesis, training and live listening.

Everything runs at openWakeWord's native format: 16 kHz mono, processed in
80 ms frames (1280 samples).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280  # 80 ms
CLIP_SAMPLES = 32_000  # 2 s
N_FRAMES = 16  # embedding frames the classifier sees (~1.3 s of context)
EMBEDDING_DIM = 96


def to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def pad_center(audio: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    """Center-crop or zero-pad to exactly ``length`` samples."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= length:
        start = (audio.size - length) // 2
        return audio[start : start + length]
    pad = length - audio.size
    return np.pad(audio, (pad // 2, pad - pad // 2))


def make_beep(freq: float = 880.0, seconds: float = 0.18, volume: float = 0.25) -> np.ndarray:
    """A short sine tone with a 5 ms fade so it does not click."""
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32) * volume
    fade = min(80, n // 2)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
    tone[:fade] *= ramp
    tone[n - fade :] *= ramp[::-1]
    return tone


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    orig_sr = int(orig_sr)
    if orig_sr == SAMPLE_RATE:
        return audio
    from scipy.signal import resample_poly

    g = math.gcd(SAMPLE_RATE, orig_sr)
    return resample_poly(audio, SAMPLE_RATE // g, orig_sr // g).astype(np.float32)


def load_wav_16k(path: str | Path) -> np.ndarray:
    """Read a WAV file as float32 mono at 16 kHz in [-1, 1]."""
    from scipy.io import wavfile

    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 1.5:
            data = data / peak
    return resample_to_16k(data, int(sr))


def write_wav_16k(path: str | Path, audio: np.ndarray) -> None:
    from scipy.io import wavfile

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SAMPLE_RATE, to_int16(audio))


class FrameChunker:
    """Turns arbitrary-length input blocks into fixed 80 ms frames.

    Live devices often run at 44.1 or 48 kHz with odd block sizes. When the
    input rate differs from 16 kHz a streaming soxr resampler keeps state
    between blocks, so frame boundaries never introduce clicks.
    """

    def __init__(self, in_sr: int, out_sr: int = SAMPLE_RATE, frame: int = FRAME_SAMPLES):
        self.in_sr = int(in_sr)
        self.out_sr = int(out_sr)
        self.frame = frame
        self._buf = np.zeros(0, dtype=np.float32)
        self._soxr = None
        if self.in_sr != self.out_sr:
            import soxr

            self._soxr = soxr.ResampleStream(self.in_sr, self.out_sr, 1, dtype="float32")

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if self._soxr is not None:
            block = np.asarray(self._soxr.resample_chunk(block), dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, block])
        n_frames = self._buf.size // self.frame
        frames = [self._buf[i * self.frame : (i + 1) * self.frame].copy() for i in range(n_frames)]
        self._buf = self._buf[n_frames * self.frame :]
        return frames
