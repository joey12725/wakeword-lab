"""openWakeWord feature extraction.

The classifier never sees raw audio. openWakeWord turns each 80 ms frame
into a 96-dim speech embedding (melspectrogram -> pretrained embedding
model), and a custom wake word is just a small model over the last 16 of
those embeddings. Training windows are extracted by streaming each clip
through the same ``AudioFeatures`` object the live ``Model`` uses, so
training and inference see identical inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import FRAME_SAMPLES, N_FRAMES, to_int16

WARMUP_FRAMES = 50


def ensure_resources() -> None:
    """Download openWakeWord's melspectrogram, embedding and VAD models once."""
    import openwakeword
    from openwakeword.utils import download_models

    needed = [
        Path(openwakeword.FEATURE_MODELS["melspectrogram"]["model_path"]).with_suffix(".onnx"),
        Path(openwakeword.FEATURE_MODELS["embedding"]["model_path"]).with_suffix(".onnx"),
        Path(openwakeword.VAD_MODELS["silero_vad"]["model_path"]),
    ]
    if not all(p.exists() for p in needed):
        # An empty list means "every pretrained wake word"; a name that matches
        # nothing fetches just the shared feature and VAD models.
        download_models(model_names=["feature-models-only"])


def make_extractor():
    from openwakeword.utils import AudioFeatures

    ensure_resources()
    return AudioFeatures(inference_framework="onnx", device="cpu")


def active_window_indices(frame_rms: np.ndarray, floor: float = 0.012, relative: float = 0.22) -> list[int]:
    """Indices of windows that end on a frame loud enough to contain speech.

    Positive clips have silence around the phrase. Labelling those silent
    windows "wake word" would teach the model to fire on silence, so only
    windows ending on frames within ``relative`` of the clip's peak energy
    are kept. Falls back to the single loudest window.
    """
    frame_rms = np.asarray(frame_rms, dtype=np.float32)
    if frame_rms.size == 0:
        return []
    cutoff = max(floor, relative * float(frame_rms.max()))
    picked = np.flatnonzero(frame_rms >= cutoff).tolist()
    return picked or [int(np.argmax(frame_rms))]


def extract_windows(audio: np.ndarray, extractor, positive: bool = False) -> list[np.ndarray]:
    """Stream a float32 16 kHz clip and collect one (16, 96) window per frame."""
    extractor.reset()
    silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    for _ in range(WARMUP_FRAMES):
        extractor(silence)

    x = np.concatenate([to_int16(audio), np.zeros(FRAME_SAMPLES * 3, dtype=np.int16)])
    windows, frame_rms = [], []
    for start in range(0, x.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        chunk = x[start : start + FRAME_SAMPLES]
        extractor(chunk)
        windows.append(extractor.get_features(N_FRAMES)[0].astype(np.float32))
        frame_rms.append(float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))))

    if not positive:
        return windows
    return [windows[i] for i in active_window_indices(np.array(frame_rms))]
