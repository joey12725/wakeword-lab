"""Interactive Hey Joy wake-word demo for this machine.

Uses Piper TTS clips plus a short microphone enrollment, trains a tiny
openWakeWord-compatible ONNX classifier, then listens live.

Run with the oww conda env:

    python demo.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CLIPS_DIR = ROOT / "clips"
MODELS_DIR = ROOT / "models"
PIPER_DIR = ROOT / "piper1-gpl"
PIPER_PY = PIPER_DIR / ".venv" / "Scripts" / "python.exe"
PIPER_VOICES = {
    "lessac": PIPER_DIR / "voices" / "en_US-lessac-medium.onnx",
    "amy": PIPER_DIR / "voices" / "en_US-amy-medium.onnx",
    "ryan": PIPER_DIR / "voices" / "en_US-ryan-high.onnx",
}

SAMPLE_RATE = 16000
CLIP_SAMPLES = 32000  # 2.0s
FRAME_SAMPLES = 1280  # 80ms
N_FRAMES = 16
WAKE_PHRASE = "Hey Joy"
MODEL_NAME = "hey_joy"

POSITIVE_TEXTS = ["Hey Joy.", "Hey, Joy.", "Hey Joy!"]
NEGATIVE_TEXTS = [
    "Hey Google.",
    "Hey Siri.",
    "Hey Jarvis.",
    "Hey Joey.",
    "Hey John.",
    "Enjoy.",
    "Hey.",
    "Joy.",
    "Hello there.",
    "Okay Google.",
    "What's up.",
    "How are you.",
    "Good morning.",
]


def _print(msg: str = "", **kwargs) -> None:
    print(msg, flush=True, **kwargs)


# ---------------------------------------------------------------------------
# Piper worker (invoked as: python demo.py --generate-tts)
# ---------------------------------------------------------------------------
def generate_tts_main() -> None:
    from piper import PiperVoice, SynthesisConfig

    out_pos = CLIPS_DIR / "positive"
    out_neg = CLIPS_DIR / "negative"
    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)

    voices = {}
    for name, path in PIPER_VOICES.items():
        if not path.exists():
            _print(f"  skip missing voice: {path.name}")
            continue
        _print(f"  loading {path.name} ...")
        voices[name] = PiperVoice.load(str(path))

    if not voices:
        raise SystemExit("No Piper voices found.")

    length_scales = [0.9, 1.0, 1.15]
    jobs = []
    for vname in voices:
        for ti, text in enumerate(POSITIVE_TEXTS):
            for ls in length_scales:
                jobs.append(("positive", vname, text, ls, out_pos / f"{vname}_pos{ti}_ls{ls:.2f}.wav"))
        for ti, text in enumerate(NEGATIVE_TEXTS):
            jobs.append(("negative", vname, text, 1.0, out_neg / f"{vname}_neg{ti}.wav"))

    remaining = [j for j in jobs if not j[4].exists()]
    _print(f"  synthesizing {len(remaining)} clips ({len(jobs) - len(remaining)} already on disk)")
    for i, (_kind, vname, text, ls, dest) in enumerate(remaining, 1):
        cfg = SynthesisConfig(length_scale=ls, noise_scale=0.7, noise_w_scale=0.8)
        with wave.open(str(dest), "wb") as wav_file:
            voices[vname].synthesize_wav(text, wav_file, syn_config=cfg)
        if i == 1 or i % 10 == 0 or i == len(remaining):
            _print(f"    {i}/{len(remaining)}  {dest.name}")

    # Spoken confirmation used when the wake word fires
    ack = CLIPS_DIR / "ack_yes.wav"
    if not ack.exists():
        voice = voices.get("lessac") or next(iter(voices.values()))
        with wave.open(str(ack), "wb") as wav_file:
            voice.synthesize_wav("Yes?", wav_file)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_wav_16k(path: Path) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
        peak = float(np.max(np.abs(data)) or 1.0)
        if peak > 1.5:
            data = data / peak
    if int(sr) != SAMPLE_RATE:
        g = math.gcd(SAMPLE_RATE, int(sr))
        data = resample_poly(data, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return data


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    from scipy.signal import resample_poly

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    orig_sr = int(orig_sr)
    if orig_sr == SAMPLE_RATE:
        return audio
    g = math.gcd(SAMPLE_RATE, orig_sr)
    return resample_poly(audio, SAMPLE_RATE // g, orig_sr // g).astype(np.float32)


class StreamResampler:
    """Stateful resampler so 48 kHz WASAPI chunks become clean 16 kHz frames."""

    def __init__(self, in_sr: int, out_sr: int = SAMPLE_RATE):
        self.in_sr = int(in_sr)
        self.out_sr = int(out_sr)
        self.out_acc = np.zeros(0, dtype=np.float32)
        self._soxr = None
        if self.in_sr != self.out_sr:
            import soxr

            self._soxr = soxr.ResampleStream(self.in_sr, self.out_sr, 1, dtype="float32")

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self._soxr is None:
            y = chunk
        else:
            y = np.asarray(self._soxr.resample_chunk(chunk), dtype=np.float32).reshape(-1)
        self.out_acc = np.concatenate([self.out_acc, y])
        frames = []
        while self.out_acc.size >= FRAME_SAMPLES:
            frames.append(self.out_acc[:FRAME_SAMPLES].copy())
            self.out_acc = self.out_acc[FRAME_SAMPLES:]
        return frames


def capture_config(device_idx: int) -> tuple[int, int]:
    """Return a (samplerate, channels) pair this input device will actually open."""
    import sounddevice as sd

    info = sd.query_devices(device_idx)
    native_sr = int(info["default_samplerate"])
    max_ch = int(info["max_input_channels"])
    channel_choices = []
    for ch in (1, 2, max_ch):
        if 1 <= ch <= max_ch and ch not in channel_choices:
            channel_choices.append(ch)
    rate_choices = []
    for sr in (SAMPLE_RATE, 48000, 44100, native_sr):
        if sr and sr not in rate_choices:
            rate_choices.append(int(sr))
    for ch in channel_choices:
        for sr in rate_choices:
            try:
                sd.check_input_settings(device=device_idx, channels=ch, samplerate=sr, dtype="float32")
                return sr, ch
            except Exception:
                continue
    return native_sr, max(1, min(max_ch, 1))


def to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def pad_center(audio: np.ndarray, length: int = CLIP_SAMPLES) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= length:
        start = (audio.size - length) // 2
        return audio[start : start + length]
    pad = length - audio.size
    return np.pad(audio, (pad // 2, pad - pad // 2))


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def make_beep(freq: float = 880.0, seconds: float = 0.18, volume: float = 0.25) -> np.ndarray:
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    env = np.linspace(0, 1, 80, endpoint=False)
    wave_ = np.sin(2 * np.pi * freq * t).astype(np.float32) * volume
    wave_[:80] *= env
    wave_[-80:] *= env[::-1]
    return wave_


def play_audio(audio: np.ndarray, sr: int = SAMPLE_RATE, output_device=None) -> None:
    import sounddevice as sd

    sd.play(np.clip(audio, -1.0, 1.0), sr, device=output_device)
    sd.wait()


def choose_input_device(requested: str | None) -> int:
    import sounddevice as sd

    devices = sd.query_devices()
    inputs = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        host = sd.query_hostapis()[dev["hostapi"]]["name"]
        inputs.append((idx, dev["name"], host, int(dev["max_input_channels"])))

    def score(item: tuple) -> tuple:
        idx, name, host, _ch = item
        n = name.lower()
        h = host.lower()
        s = 0
        if "wasapi" in h:
            s += 8
        elif "mme" in h:
            s += 3
        if "umc" in n:
            s += 30
        if "anker" in n or "powerconf" in n:
            s += 20
        if "falcon" in n or "creality" in n:
            s -= 40
        if "steam" in n:
            s -= 40
        if "mapper" in n or "primary" in n:
            s -= 20
        return (s, -idx)

    if requested:
        if requested.isdigit():
            idx = int(requested)
            if devices[idx]["max_input_channels"] <= 0:
                raise SystemExit(f"Device {idx} is not an input device.")
            return idx
        needle = requested.lower()
        matches = [i for i in inputs if needle in i[1].lower()]
        if not matches:
            raise SystemExit(f"No input device matching '{requested}'. Use --list-devices.")
        matches.sort(key=score, reverse=True)
        return matches[0][0]

    ranked = sorted(inputs, key=score, reverse=True)
    return ranked[0][0]


def list_devices() -> None:
    import sounddevice as sd

    _print("Audio devices:")
    for idx, dev in enumerate(sd.query_devices()):
        host = sd.query_hostapis()[dev["hostapi"]]["name"]
        kind = []
        if dev["max_input_channels"]:
            kind.append(f"{dev['max_input_channels']} in")
        if dev["max_output_channels"]:
            kind.append(f"{dev['max_output_channels']} out")
        mark = ""
        if idx == sd.default.device[0]:
            mark += "  [default in]"
        if idx == sd.default.device[1]:
            mark += "  [default out]"
        _print(f"  {idx:3d}  {dev['name']}  ({host}, {', '.join(kind)}){mark}")


# ---------------------------------------------------------------------------
# Features + training
# ---------------------------------------------------------------------------
def extract_windows(audio_f32: np.ndarray, features=None, positive: bool = False) -> list[np.ndarray]:
    """Take openWakeWord streaming embeddings, matching live Model.predict()."""
    from openwakeword.utils import AudioFeatures

    feat = features or AudioFeatures(inference_framework="onnx", device="cpu")
    feat.reset()
    silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    for _ in range(50):
        feat(silence)
    x = to_int16(audio_f32)
    x = np.concatenate([x, np.zeros(FRAME_SAMPLES * 3, dtype=np.int16)])
    windows = []
    rmses = []
    for i in range(0, x.size - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        chunk = x[i : i + FRAME_SAMPLES]
        feat(chunk)
        windows.append(feat.get_features(N_FRAMES)[0].astype(np.float32))
        rmses.append(float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2))))
    if not windows:
        return []
    rmses = np.array(rmses, dtype=np.float32)
    if not positive:
        return windows
    peak = float(rmses.max()) if rmses.size else 0.0
    cutoff = max(0.012, 0.22 * peak)
    picked = [w for w, r in zip(windows, rmses) if r >= cutoff]
    if picked:
        return picked
    return [windows[int(np.argmax(rmses))]]


def augment_clip(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = audio * float(rng.uniform(0.55, 1.35))
    if rng.random() < 0.7:
        noise = rng.normal(0, rng.uniform(0.004, 0.03), out.shape).astype(np.float32)
        out = out + noise
    if rng.random() < 0.4:
        shift = int(rng.integers(-2400, 2400))
        out = np.roll(out, shift)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def gather_training_data(features, extra_pos: list[np.ndarray], extra_neg: list[np.ndarray]):
    rng = np.random.default_rng(7)
    X: list[np.ndarray] = []
    y: list[int] = []

    pos_files = sorted((CLIPS_DIR / "positive").glob("*.wav")) if (CLIPS_DIR / "positive").exists() else []
    neg_files = sorted((CLIPS_DIR / "negative").glob("*.wav")) if (CLIPS_DIR / "negative").exists() else []

    def add_clip(audio: np.ndarray, label: int, copies: int) -> None:
        variants = [audio] + [augment_clip(audio, rng) for _ in range(copies)]
        for v in variants:
            for w in extract_windows(v, features, positive=(label == 1)):
                X.append(w)
                y.append(label)

    for path in pos_files:
        add_clip(load_wav_16k(path), 1, copies=2)
    for path in neg_files:
        name = path.name.lower()
        copies = 4 if any(s in name for s in ("hey_google", "hey_joey", "hey_jarvis", "hey_siri", "hey_john", "_hey.")) else 1
        add_clip(load_wav_16k(path), 0, copies=copies)

    for audio in extra_pos:
        add_clip(audio, 1, copies=10)
    for audio in extra_neg:
        # slice room audio into 2s hops
        if audio.size > CLIP_SAMPLES:
            hop = SAMPLE_RATE // 2
            for start in range(0, audio.size - CLIP_SAMPLES + 1, hop):
                add_clip(audio[start : start + CLIP_SAMPLES], 0, copies=1)
        else:
            add_clip(audio, 0, copies=2)

    # Pure noise / silence negatives so the model does not fire in a quiet room
    for _ in range(12):
        add_clip(rng.normal(0, rng.uniform(0.002, 0.02), CLIP_SAMPLES).astype(np.float32), 0, copies=0)
    add_clip(np.zeros(CLIP_SAMPLES, dtype=np.float32), 0, copies=0)

    if not X:
        raise SystemExit("No training clips found. Generate TTS samples first.")
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32)


def train_onnx(X: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    import torch
    from torch import nn

    class WakeNet(nn.Module):
        def __init__(self, dim: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(N_FRAMES * 96, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WakeNet().to(device)
    xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device).unsqueeze(1)

    n_pos = max(1, int((y == 1).sum()))
    n_neg = max(1, int((y == 0).sum()))
    # Wake-word models should be conservative: punish false positives harder.
    pos_w = torch.tensor([1.0], device=device)
    neg_w = torch.tensor([2.2], device=device)

    opt = torch.optim.Adam(model.parameters(), lr=8e-4)
    loss_fn = nn.BCELoss(reduction="none")
    n = xt.shape[0]
    batch = min(64, n)
    epochs = 35 if device.type == "cuda" else 22

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            pred = model(xt[idx])
            raw = loss_fn(pred, yt[idx])
            w = torch.where(yt[idx] > 0.5, pos_w, neg_w)
            loss = (raw * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.shape[0]
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 5 == 0:
            with torch.no_grad():
                pred = model(xt)
                p = (pred.squeeze() >= 0.5).float()
                acc = float((p == yt.squeeze()).float().mean())
            _print(f"    epoch {epoch + 1:02d}/{epochs}  loss={total / n:.4f}  acc={acc:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, N_FRAMES, 96)
    torch.onnx.export(
        model.cpu(),
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=[MODEL_NAME],
        opset_version=13,
    )
    _print(f"  saved {out_path}")


def evaluate_model(model_path: Path) -> None:
    from openwakeword.model import Model

    m = Model(wakeword_models=[str(model_path)], inference_framework="onnx", vad_threshold=0.0)

    def peak(path: Path) -> float:
        m.reset()
        for _ in range(50):
            m.predict(np.zeros(FRAME_SAMPLES, dtype=np.int16))
        audio = to_int16(load_wav_16k(path))
        preds = m.predict_clip(audio, padding=0)
        scores = [float(p[MODEL_NAME]) for p in preds]
        return max(scores) if scores else 0.0

    _print("  streaming check (should be high on Hey Joy, low on other speech):")
    samples = [
        ("enroll", CLIPS_DIR / "enroll" / "pos_01.wav"),
        ("enroll", CLIPS_DIR / "enroll" / "pos_03.wav"),
        ("tts hey joy", next((CLIPS_DIR / "positive").glob("*.wav"))),
        ("hey google", CLIPS_DIR / "negative" / "lessac_hey_google.wav"),
        ("hello there", CLIPS_DIR / "negative" / "lessac_hello_there.wav"),
        ("room", CLIPS_DIR / "enroll" / "neg_room.wav"),
    ]
    for label, path in samples:
        if path and path.exists():
            _print(f"    {label:12}  {peak(path):.3f}  {path.name}")


# ---------------------------------------------------------------------------
# Enrollment + live listen
# ---------------------------------------------------------------------------
def record_seconds(seconds: float, input_device: int, channel: int = 0) -> np.ndarray:
    import sounddevice as sd

    sr, channels = capture_config(input_device)
    channel = min(max(0, channel), channels - 1)
    frames = int(seconds * sr)
    audio = sd.rec(
        frames,
        samplerate=sr,
        channels=channels,
        dtype="float32",
        device=input_device,
    )
    sd.wait()
    if audio.ndim > 1:
        audio = audio[:, channel]
    return resample_to_16k(audio.reshape(-1).astype(np.float32), sr)


def probe_best_mic(preferred_device: int | None = None, preferred_channel: int | None = None) -> tuple[int, int]:
    """Listen briefly and pick the input device/channel with the most energy."""
    import sounddevice as sd

    if preferred_device is not None and preferred_channel is not None:
        return preferred_device, preferred_channel

    devices = sd.query_devices()
    candidates = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        host = sd.query_hostapis()[dev["hostapi"]]["name"].lower()
        name = dev["name"].lower()
        if any(bad in name for bad in ("steam", "falcon", "creality", "mapper", "primary")):
            continue
        if "wdm-ks" in host:
            continue
        if not any(h in host for h in ("wasapi", "mme", "directsound")):
            continue
        candidates.append(idx)

    if preferred_device is not None:
        candidates = [preferred_device]
    elif not candidates:
        raise SystemExit("No usable input devices found. Use --list-devices.")

    _print("  Mic check: talk, clap, or make noise until the device list finishes...")
    play_audio(make_beep(freq=660.0, seconds=0.12))
    ranked = []
    for idx in candidates:
        sr, channels = capture_config(idx)
        frames = int(1.1 * sr)
        raw = sd.rec(frames, samplerate=sr, channels=channels, dtype="float32", device=idx)
        sd.wait()
        name = devices[idx]["name"]
        ch_count = 1 if raw.ndim == 1 else raw.shape[1]
        for ch in range(ch_count):
            col = raw if raw.ndim == 1 else raw[:, ch]
            energy = rms(resample_to_16k(col.reshape(-1).astype(np.float32), sr))
            ranked.append((energy, idx, ch, name))
            _print(f"    device {idx} ch{ch}  rms={energy:.4f}  {name}")

    ranked.sort(reverse=True)
    if preferred_channel is not None and preferred_device is not None:
        return preferred_device, preferred_channel

    def supports_16k(idx: int) -> bool:
        try:
            sd.check_input_settings(device=idx, channels=1, samplerate=SAMPLE_RATE, dtype="float32")
            return True
        except Exception:
            return False

    best_e = ranked[0][0]
    eligible = [r for r in ranked if r[0] >= max(1e-6, 0.4 * best_e)]
    sixteen = [r for r in eligible if supports_16k(r[1])]
    best_e, best_idx, best_ch, best_name = (sixteen[0] if sixteen else ranked[0])
    if best_e < 0.004:
        _print("  warning: all mics were very quiet. Using the loudest anyway.")
        _print("  If this is wrong, rerun with --list-devices and --device / --channel")
    cap_sr, _cap_ch = capture_config(best_idx)
    _print(f"  selected device {best_idx} ch{best_ch} ({best_name}), rms={best_e:.4f}, open@{cap_sr}Hz")
    return best_idx, best_ch


def enroll(input_device: int, channel: int = 0, n: int = 5) -> tuple[list[np.ndarray], np.ndarray]:
    _print("")
    _print("  Enrollment teaches the detector YOUR voice on this mic.")
    _print(f"  After the countdown, a beep plays. Then say:  {WAKE_PHRASE}")
    for remaining in range(3, 0, -1):
        _print(f"  starting in {remaining}...")
        time.sleep(1)

    positives: list[np.ndarray] = []
    for i in range(1, n + 1):
        while True:
            _print(f"  [{i}/{n}] Get ready...")
            time.sleep(0.35)
            play_audio(make_beep())
            _print(f"      say  {WAKE_PHRASE}")
            clip = record_seconds(1.8, input_device, channel)
            energy = rms(clip)
            if energy < 0.008:
                _print(f"      too quiet (rms={energy:.4f}). Try closer to the mic.")
                continue
            positives.append(clip)
            _print(f"      captured  rms={energy:.3f}")
            break

    _print("")
    _print("  Now collect negatives: stay quiet, or talk about something else.")
    _print("  Do NOT say the wake word. Recording 6 seconds...")
    time.sleep(0.6)
    play_audio(make_beep(freq=440.0))
    negatives = record_seconds(6.0, input_device, channel)
    _print(f"  room/other speech captured  rms={rms(negatives):.3f}")
    return positives, negatives


def listen_loop(model_path: Path, input_device: int, channel: int, threshold: float, debounce: float) -> None:
    import sounddevice as sd
    import openwakeword
    from openwakeword.model import Model

    vad_path = Path(openwakeword.VAD_MODELS["silero_vad"]["model_path"])
    if not vad_path.exists():
        openwakeword.utils.download_models(model_names=[])

    oww = Model(
        wakeword_models=[str(model_path)],
        inference_framework="onnx",
        vad_threshold=0.35,
    )

    ack_path = CLIPS_DIR / "ack_yes.wav"
    ack = load_wav_16k(ack_path) if ack_path.exists() else make_beep(freq=1320.0, seconds=0.25)

    dev = sd.query_devices(input_device)
    _print("")
    _print("=" * 62)
    _print(f"  Listening for '{WAKE_PHRASE}'")
    _print(f"  mic        {dev['name']}  ch{channel}")
    _print(f"  model      {model_path.name}")
    _print(f"  threshold  {threshold:.2f}   (raise if it false-triggers)")
    _print("  Ctrl+C to quit")
    _print("=" * 62)
    _print("")

    hits = 0
    armed = True
    ignore_until = 0.0
    cap_sr, cap_ch = capture_config(input_device)
    native_block = max(1, int(round(cap_sr * FRAME_SAMPLES / SAMPLE_RATE)))
    resampler = StreamResampler(cap_sr, SAMPLE_RATE)
    _print(f"  capture    {cap_sr} Hz, {cap_ch} ch -> 16 kHz streaming resample")
    # Flush the random feature-buffer so live frames match training.
    for _ in range(50):
        oww.predict(np.zeros(FRAME_SAMPLES, dtype=np.int16))
    try:
        with sd.InputStream(
            samplerate=cap_sr,
            channels=cap_ch,
            dtype="float32",
            blocksize=native_block,
            device=input_device,
        ) as stream:
            while True:
                block, overflowed = stream.read(native_block)
                if block.ndim > 1:
                    block = block[:, min(channel, block.shape[1] - 1)]
                for audio_16k in resampler.push(block.reshape(-1).astype(np.float32)):
                    audio_i16 = to_int16(audio_16k)
                    scores = oww.predict(audio_i16)
                    score = float(scores.get(MODEL_NAME, 0.0))
                    energy = rms(audio_16k)
                    width = 28
                    filled = max(0, min(width, int(round(score * width))))
                    bar = "#" * filled + "-" * (width - filled)
                    flag = ""
                    now = time.time()
                    if now < ignore_until:
                        continue
                    if armed and score >= threshold:
                        hits += 1
                        armed = False
                        ignore_until = now + max(debounce, 2.2)
                        flag = f"  *** {WAKE_PHRASE.upper()}  #{hits} ***"
                        _print(f"  {score:5.2f}  rms={energy:.3f}  [{bar}]{flag}")
                        try:
                            play_audio(ack)
                        except Exception:
                            pass
                    else:
                        if score < threshold * 0.4:
                            armed = True
                        _print(f"  {score:5.2f}  rms={energy:.3f}  [{bar}]{flag}   ", end="\r")
    except KeyboardInterrupt:
        _print("\n\nStopped.")
        _print(f"Detections this session: {hits}")


def ensure_tts_clips() -> None:
    pos = CLIPS_DIR / "positive"
    if pos.exists() and any(pos.glob("*.wav")):
        n = len(list(pos.glob("*.wav")))
        _print(f"  using {n} existing Piper clips in {pos}")
        return
    _print("  generating Piper TTS clips (first run only)...")
    try:
        generate_tts_main()
    except Exception as exc:
        _print(f"  warning: Piper generation failed ({exc}); demo will rely on enrollment.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Hey Joy wake-word demo")
    parser.add_argument("--generate-tts", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", default=None, help="Input device index or name substring")
    parser.add_argument("--channel", type=int, default=None, help="Input channel index for multi-channel interfaces")
    parser.add_argument("--skip-enroll", action="store_true", help="Skip microphone enrollment")
    parser.add_argument("--train-only", action="store_true", help="Train the model and exit")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if a model exists")
    parser.add_argument("--threshold", type=float, default=0.65, help="Detection threshold (0-1)")
    parser.add_argument("--enroll-count", type=int, default=5, help="Enrollment repetitions")
    args = parser.parse_args()

    if args.generate_tts:
        generate_tts_main()
        return

    if args.list_devices:
        list_devices()
        return

    import openwakeword
    from openwakeword.utils import AudioFeatures

    _print("=" * 62)
    _print("  Hey Joy  -  live wake-word demo")
    _print("=" * 62)

    input_device = choose_input_device(args.device) if args.device else None
    import sounddevice as sd

    _print("\n[1/4] Microphone")
    input_device, channel = probe_best_mic(preferred_device=input_device, preferred_channel=args.channel)
    dev = sd.query_devices(input_device)
    _print(f"  using {dev['name']}  (device {input_device}, channel {channel})")
    _print("  override with --device <index> --channel <n>   or  --list-devices")

    _print("\n[2/4] Reference clips")
    ensure_tts_clips()

    extra_pos: list[np.ndarray] = []
    extra_neg: list[np.ndarray] = []
    enroll_dir = CLIPS_DIR / "enroll"
    model_path = MODELS_DIR / f"{MODEL_NAME}.onnx"

    if not args.skip_enroll:
        _print("\n[3/4] Voice enrollment")
        extra_pos, extra_neg_audio = enroll(input_device, channel=channel, n=args.enroll_count)
        extra_neg = [extra_neg_audio]
        enroll_dir.mkdir(parents=True, exist_ok=True)
        for i, clip in enumerate(extra_pos, 1):
            from scipy.io import wavfile

            wavfile.write(enroll_dir / f"pos_{i:02d}.wav", SAMPLE_RATE, to_int16(clip))
        from scipy.io import wavfile

        wavfile.write(enroll_dir / "neg_room.wav", SAMPLE_RATE, to_int16(extra_neg_audio))
        args.retrain = True
    else:
        _print("\n[3/4] Skipping enrollment")
        if enroll_dir.exists():
            extra_pos = [load_wav_16k(p) for p in sorted(enroll_dir.glob("pos_*.wav"))]
            neg_path = enroll_dir / "neg_room.wav"
            if neg_path.exists():
                extra_neg = [load_wav_16k(neg_path)]

    need_train = args.retrain or not model_path.exists()
    _print("\n[4/4] Detector")
    if need_train:
        _print("  extracting openWakeWord embeddings and training...")
        if not Path(openwakeword.VAD_MODELS["silero_vad"]["model_path"]).exists():
            openwakeword.utils.download_models(model_names=[])
        features = AudioFeatures(inference_framework="onnx", device="cpu")
        X, y = gather_training_data(features, extra_pos, extra_neg)
        _print(f"  windows: {len(y)}  (pos={(y == 1).sum()}  neg={(y == 0).sum()})")
        train_onnx(X, y, model_path)
        evaluate_model(model_path)
    else:
        _print(f"  using existing {model_path}")

    if args.train_only:
        _print("Train-only complete.")
        return

    listen_loop(model_path, input_device, channel, threshold=args.threshold, debounce=1.6)


if __name__ == "__main__":
    main()
