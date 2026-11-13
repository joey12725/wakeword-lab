"""Microphone discovery: list devices, pick a sensible default, and probe
which device/channel is actually hearing you."""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE, make_beep, resample_to_16k, rms

# Virtual, loopback and "system mapper" inputs rarely carry a real mic.
_VIRTUAL_HINTS = ("steam", "mapper", "primary sound", "virtual", "loopback", "stereo mix", "blackhole", "zoomaudio")
_HOST_SCORES = {"wasapi": 8, "core audio": 8, "alsa": 4, "pulse": 6, "mme": 3, "directsound": 2, "wdm-ks": -10}


def score_device(name: str, host: str) -> int:
    """Higher is a better default input. Pure, so it is easy to test."""
    n, h = name.lower(), host.lower()
    score = next((v for k, v in _HOST_SCORES.items() if k in h), 0)
    if any(hint in n for hint in _VIRTUAL_HINTS):
        score -= 40
    if "mic" in n or "microphone" in n:
        score += 5
    return score


def input_devices() -> list[tuple[int, str, str, int]]:
    import sounddevice as sd

    hosts = sd.query_hostapis()
    return [
        (idx, dev["name"], hosts[dev["hostapi"]]["name"], int(dev["max_input_channels"]))
        for idx, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] > 0
    ]


def choose_input_device(requested: str | None = None) -> int:
    """Resolve ``--device`` (index or name substring) or pick the best default."""
    inputs = input_devices()
    if not inputs:
        raise SystemExit("No audio input devices found.")
    ranked = sorted(inputs, key=lambda d: (score_device(d[1], d[2]), -d[0]), reverse=True)
    if not requested:
        return ranked[0][0]
    if requested.isdigit():
        idx = int(requested)
        if idx not in {d[0] for d in inputs}:
            raise SystemExit(f"Device {idx} is not an input device. Use `wakeword-lab devices`.")
        return idx
    matches = [d for d in ranked if requested.lower() in d[1].lower()]
    if not matches:
        raise SystemExit(f"No input device matching '{requested}'. Use `wakeword-lab devices`.")
    return matches[0][0]


def list_devices(log=print) -> None:
    import sounddevice as sd

    default_in, default_out = sd.default.device
    log("Audio devices:")
    for idx, dev in enumerate(sd.query_devices()):
        host = sd.query_hostapis()[dev["hostapi"]]["name"]
        kind = []
        if dev["max_input_channels"]:
            kind.append(f"{dev['max_input_channels']} in")
        if dev["max_output_channels"]:
            kind.append(f"{dev['max_output_channels']} out")
        mark = ("  [default in]" if idx == default_in else "") + ("  [default out]" if idx == default_out else "")
        log(f"  {idx:3d}  {dev['name']}  ({host}, {', '.join(kind)}){mark}")


def capture_config(device: int) -> tuple[int, int]:
    """Return a (samplerate, channels) pair this input device will open.

    Prefers 16 kHz mono so no resampling is needed, then common rates.
    """
    import sounddevice as sd

    info = sd.query_devices(device)
    native_sr = int(info["default_samplerate"])
    max_ch = int(info["max_input_channels"])
    channels = list(dict.fromkeys(c for c in (1, 2, max_ch) if 1 <= c <= max_ch))
    rates = list(dict.fromkeys((SAMPLE_RATE, 48_000, 44_100, native_sr)))
    for ch in channels:
        for sr in rates:
            try:
                sd.check_input_settings(device=device, channels=ch, samplerate=sr, dtype="float32")
                return sr, ch
            except Exception:
                continue
    return native_sr, 1


def record(seconds: float, device: int, channel: int = 0) -> np.ndarray:
    """Record from one channel and return float32 audio at 16 kHz."""
    import sounddevice as sd

    sr, channels = capture_config(device)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=channels, dtype="float32", device=device)
    sd.wait()
    if audio.ndim > 1:
        audio = audio[:, min(max(0, channel), audio.shape[1] - 1)]
    return resample_to_16k(audio.reshape(-1), sr)


def play(audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    import sounddevice as sd

    sd.play(np.clip(audio, -1.0, 1.0), sr)
    sd.wait()


def probe_loudest(device: int | None = None, log=print) -> tuple[int, int]:
    """Record ~1 s from each candidate input and return the loudest (device, channel).

    Handy on multi-channel audio interfaces where the mic is on input 2 and
    nobody remembers which index the OS assigned.
    """
    import sounddevice as sd

    candidates = (
        [device]
        if device is not None
        else [idx for idx, name, host, _ in input_devices() if score_device(name, host) > -10]
    )
    if not candidates:
        raise SystemExit("No usable input devices found. Use `wakeword-lab devices`.")

    log("  Mic check: talk or clap until the list finishes...")
    play(make_beep(freq=660.0, seconds=0.12))
    ranked = []
    for idx in candidates:
        sr, channels = capture_config(idx)
        raw = sd.rec(int(1.1 * sr), samplerate=sr, channels=channels, dtype="float32", device=idx)
        sd.wait()
        raw = raw.reshape(len(raw), -1)
        for ch in range(raw.shape[1]):
            energy = rms(resample_to_16k(raw[:, ch], sr))
            ranked.append((energy, idx, ch))
            log(f"    device {idx} ch{ch}  rms={energy:.4f}  {sd.query_devices(idx)['name']}")

    energy, idx, ch = max(ranked)
    if energy < 0.004:
        log("  every input was very quiet; using the loudest anyway (override with --device/--channel)")
    log(f"  selected device {idx} channel {ch}")
    return idx, ch
