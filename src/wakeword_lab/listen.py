"""Live detection and optional voice enrollment."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .audio import FRAME_SAMPLES, SAMPLE_RATE, FrameChunker, load_wav_16k, make_beep, rms, to_int16, write_wav_16k
from .config import WakeWordConfig
from .devices import capture_config, play, record
from .features import WARMUP_FRAMES, ensure_resources


@dataclass
class TriggerGate:
    """Turns a per-frame score stream into discrete detections.

    Fires when the score crosses ``threshold``, then stays quiet for
    ``refractory`` seconds and until the score drops below
    ``threshold * rearm_ratio``, so one utterance never fires twice.
    """

    threshold: float
    refractory: float = 2.2
    rearm_ratio: float = 0.4
    armed: bool = field(default=True, init=False)
    quiet_until: float = field(default=0.0, init=False)

    def update(self, score: float, now: float) -> bool:
        if now < self.quiet_until:
            return False
        if self.armed and score >= self.threshold:
            self.armed = False
            self.quiet_until = now + self.refractory
            return True
        if score < self.threshold * self.rearm_ratio:
            self.armed = True
        return False


def meter(score: float, width: int = 28) -> str:
    filled = max(0, min(width, int(round(score * width))))
    return "#" * filled + "-" * (width - filled)


def enroll(config: WakeWordConfig, device: int, channel: int, takes: int = 5, log=print):
    """Record a few takes of the wake phrase plus some room tone.

    Takes are saved under the run's ``enroll/`` folder (never committed) and
    reused on later trainings.
    """
    phrase = config.phrase
    log(f"\n  After each beep, say: {phrase}")
    for remaining in (3, 2, 1):
        log(f"  starting in {remaining}...")
        time.sleep(1)

    positives = []
    while len(positives) < takes:
        log(f"  [{len(positives) + 1}/{takes}] say  {phrase}")
        time.sleep(0.35)
        play(make_beep())
        clip = record(1.8, device, channel)
        if rms(clip) < 0.008:
            log(f"      too quiet (rms={rms(clip):.4f}); move closer to the mic")
            continue
        positives.append(clip)

    log("\n  Now 6 seconds of anything EXCEPT the wake phrase (talk, or stay quiet)...")
    time.sleep(0.6)
    play(make_beep(freq=440.0))
    room = record(6.0, device, channel)

    for i, clip in enumerate(positives, 1):
        write_wav_16k(config.enroll_dir / f"pos_{i:02d}.wav", clip)
    write_wav_16k(config.enroll_dir / "room.wav", room)
    return positives, [room]


def load_enrollment(config: WakeWordConfig) -> tuple[list[np.ndarray], list[np.ndarray]]:
    d = config.enroll_dir
    if not d.exists():
        return [], []
    pos = [load_wav_16k(p) for p in sorted(d.glob("pos_*.wav"))]
    neg = [load_wav_16k(p) for p in sorted(d.glob("room*.wav"))]
    return pos, neg


def listen(config: WakeWordConfig, device: int, channel: int, threshold: float | None = None, log=print) -> int:
    import sounddevice as sd
    from openwakeword.model import Model

    if not config.model_path.exists():
        raise SystemExit(f"No model at {config.model_path}. Run `wakeword-lab train` first.")
    ensure_resources()
    threshold = config.listen.threshold if threshold is None else threshold
    model = Model(
        wakeword_models=[str(config.model_path)],
        inference_framework="onnx",
        vad_threshold=config.listen.vad_threshold,
    )
    ack_path = config.clips_dir / "ack.wav"
    ack = load_wav_16k(ack_path) if ack_path.exists() else make_beep(freq=1320.0, seconds=0.25)

    cap_sr, cap_ch = capture_config(device)
    block = max(1, round(cap_sr * FRAME_SAMPLES / SAMPLE_RATE))
    chunker = FrameChunker(cap_sr)
    gate = TriggerGate(threshold, refractory=config.listen.refractory_seconds)

    log("")
    log("=" * 62)
    log(f"  Listening for '{config.phrase}'")
    log(f"  mic        {sd.query_devices(device)['name']}  ch{channel}  ({cap_sr} Hz)")
    log(f"  model      {config.model_path.name}")
    log(f"  threshold  {threshold:.2f}  (raise it if it false-triggers)")
    log("  Ctrl+C to quit")
    log("=" * 62)

    for _ in range(WARMUP_FRAMES):  # flush the feature buffer so live frames match training
        model.predict(np.zeros(FRAME_SAMPLES, dtype=np.int16))

    hits = 0
    try:
        with sd.InputStream(
            samplerate=cap_sr, channels=cap_ch, dtype="float32", blocksize=block, device=device
        ) as stream:
            while True:
                data, _overflowed = stream.read(block)
                if data.ndim > 1:
                    data = data[:, min(channel, data.shape[1] - 1)]
                for frame in chunker.push(data):
                    score = float(model.predict(to_int16(frame)).get(config.model_name, 0.0))
                    line = f"  {score:5.2f}  rms={rms(frame):.3f}  [{meter(score)}]"
                    if gate.update(score, time.monotonic()):
                        hits += 1
                        log(f"{line}  *** {config.phrase.upper()} #{hits} ***")
                        try:
                            play(ack)
                        except Exception:
                            pass
                    else:
                        log(line + "   ", end="\r")
    except KeyboardInterrupt:
        log(f"\n\nStopped. Detections this session: {hits}")
    return hits
