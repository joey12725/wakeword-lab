"""Synthetic training data with Piper TTS.

Every clip is planned up front as a ``ClipJob``: which text, which voice and
speaker, and which prosody settings. The plan is deterministic, so a run can
be interrupted and resumed, and a manifest records what each WAV contains.

Multi-speaker Piper voices such as ``en_US-libritts_r-medium`` (900+
speakers) give thousands of distinct voices from one small model, which is
what makes a few-thousand-clip dataset possible without any recordings.
"""

from __future__ import annotations

import csv
import itertools
import random
import wave
from dataclasses import dataclass, replace
from pathlib import Path

from .config import PhraseSpec, WakeWordConfig, slugify

MANIFEST_NAME = "manifest.csv"
MANIFEST_FIELDS = (
    "path",
    "label",
    "text",
    "weight",
    "voice",
    "speaker",
    "length_scale",
    "noise_scale",
    "noise_w_scale",
)


@dataclass(frozen=True)
class ClipJob:
    path: str  # relative to the clips directory
    label: int
    text: str
    weight: int
    voice: str
    speaker: int | None
    length_scale: float
    noise_scale: float
    noise_w_scale: float


def plan_phrase(
    spec: PhraseSpec,
    label: int,
    index: int,
    config: WakeWordConfig,
    speaker_counts: dict[str, int],
) -> list[ClipJob]:
    """Spread ``spec.clips`` clips across every voice/speaker/prosody combo.

    The full grid is shuffled with a per-phrase seed and walked in order, so
    small clip counts still cover many voices and large counts only repeat a
    combination once the grid is exhausted.
    """
    syn = config.synthesis
    speakers: list[tuple[str, int | None]] = []
    for voice in syn.voices:
        n = int(speaker_counts.get(voice, 1))
        if n <= 1:
            speakers.append((voice, None))
            continue
        if syn.max_speakers_per_voice:
            n = min(n, syn.max_speakers_per_voice)
        speakers.extend((voice, s) for s in range(n))

    grid = list(itertools.product(speakers, syn.length_scales, syn.noise_scales, syn.noise_w_scales))
    random.Random(f"{syn.seed}:{label}:{spec.text}").shuffle(grid)

    kind = "positive" if label else "negative"
    folder = f"{kind}/{index:02d}_{slugify(spec.text)}"
    jobs = []
    for i in range(spec.clips):
        (voice, speaker), ls, ns, nw = grid[i % len(grid)]
        spk = "" if speaker is None else f"_s{speaker}"
        jobs.append(
            ClipJob(
                path=f"{folder}/{i:05d}_{voice}{spk}.wav",
                label=label,
                text=spec.text,
                weight=spec.weight,
                voice=voice,
                speaker=speaker,
                length_scale=ls,
                noise_scale=ns,
                noise_w_scale=nw,
            )
        )
    return jobs


def plan_clips(config: WakeWordConfig, speaker_counts: dict[str, int] | None = None) -> list[ClipJob]:
    speaker_counts = speaker_counts or {}
    jobs: list[ClipJob] = []
    for i, spec in enumerate(config.positives):
        jobs += plan_phrase(spec, 1, i, config, speaker_counts)
    for i, spec in enumerate(config.negatives):
        jobs += plan_phrase(spec, 0, i, config, speaker_counts)
    return jobs


def write_manifest(clips_dir: Path, jobs: list[ClipJob]) -> Path:
    path = clips_dir / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for job in jobs:
            row = {k: getattr(job, k) for k in MANIFEST_FIELDS}
            row["speaker"] = "" if job.speaker is None else job.speaker
            writer.writerow(row)
    return path


def read_manifest(clips_dir: Path) -> list[ClipJob]:
    path = clips_dir / MANIFEST_NAME
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [
            ClipJob(
                path=row["path"],
                label=int(row["label"]),
                text=row["text"],
                weight=int(row["weight"]),
                voice=row["voice"],
                speaker=int(row["speaker"]) if row["speaker"] else None,
                length_scale=float(row["length_scale"]),
                noise_scale=float(row["noise_scale"]),
                noise_w_scale=float(row["noise_w_scale"]),
            )
            for row in csv.DictReader(f)
        ]


def _load_voices(config: WakeWordConfig, log=print) -> dict:
    from piper import PiperVoice

    voices = {}
    for name in config.synthesis.voices:
        model = config.synthesis.voices_dir / f"{name}.onnx"
        if not model.exists():
            log(
                f"  missing voice {model} (download with: python -m piper.download_voices "
                f"--download-dir {config.synthesis.voices_dir} {name})"
            )
            continue
        voices[name] = PiperVoice.load(str(model))
        log(f"  loaded {name} ({voices[name].config.num_speakers} speaker(s))")
    if not voices:
        raise SystemExit("No Piper voices available. See the README for the voice download command.")
    return voices


def synthesize(config: WakeWordConfig, log=print) -> list[ClipJob]:
    """Generate every planned clip that is not already on disk."""
    from piper import SynthesisConfig

    voices = _load_voices(config, log)
    counts = {name: v.config.num_speakers for name, v in voices.items()}
    # Plan over the voices that actually loaded so every phrase still gets its full clip count.
    available = replace(config, synthesis=replace(config.synthesis, voices=tuple(voices)))
    jobs = plan_clips(available, counts)
    clips_dir = config.clips_dir
    write_manifest(clips_dir, jobs)

    todo = [j for j in jobs if not (clips_dir / j.path).exists()]
    log(f"  {len(jobs)} clips planned, {len(jobs) - len(todo)} already on disk")
    for i, job in enumerate(todo, 1):
        dest = clips_dir / job.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        syn = SynthesisConfig(
            speaker_id=job.speaker,
            length_scale=job.length_scale,
            noise_scale=job.noise_scale,
            noise_w_scale=job.noise_w_scale,
        )
        with wave.open(str(dest), "wb") as wav_file:
            voices[job.voice].synthesize_wav(job.text, wav_file, syn_config=syn)
        if i == 1 or i % 50 == 0 or i == len(todo):
            log(f"    {i}/{len(todo)}  {job.path}")

    ack = clips_dir / "ack.wav"
    if not ack.exists():
        voice = next(iter(voices.values()))
        with wave.open(str(ack), "wb") as wav_file:
            voice.synthesize_wav(config.listen.ack_text, wav_file)
    return jobs
