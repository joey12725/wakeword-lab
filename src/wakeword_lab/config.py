"""Wake-word project configuration, loaded from a TOML file.

A config describes one wake word end to end: the phrase, which texts count
as positives, which confusable phrases the model must reject, how Piper
should vary the synthetic voices, and how to train and listen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


class ConfigError(ValueError):
    """Raised when a config file is missing fields or has bad values."""


@dataclass(frozen=True)
class PhraseSpec:
    """A text to synthesize and how much it should count during training.

    ``weight`` is the number of extra augmented copies made of every clip.
    Raise it for hard negatives ("Hey Joey" vs "Hey Joy") so the classifier
    sees them more often than easy ones.
    """

    text: str
    clips: int
    weight: int = 1


@dataclass(frozen=True)
class SynthesisSettings:
    voices: tuple[str, ...] = ("en_US-lessac-medium", "en_US-amy-medium", "en_US-ryan-high")
    voices_dir: Path = Path("voices")
    length_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    noise_scales: tuple[float, ...] = (0.5, 0.667, 0.8)
    noise_w_scales: tuple[float, ...] = (0.6, 0.8, 1.0)
    max_speakers_per_voice: int = 0  # 0 = use every speaker a multi-speaker voice has
    seed: int = 1234


@dataclass(frozen=True)
class BackgroundSettings:
    """Real-world audio (music, noise, speech) sliced into negative windows."""

    dirs: tuple[Path, ...] = ()
    max_clips: int = 2000
    hop_seconds: float = 1.0


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 30
    hidden_dim: int = 64
    learning_rate: float = 8e-4
    batch_size: int = 64
    negative_weight: float = 2.2
    positive_copies: int = 2
    noise_clips: int = 12
    holdout_fraction: float = 0.15
    patience: int = 8
    seed: int = 7


@dataclass(frozen=True)
class ListenSettings:
    threshold: float = 0.65
    vad_threshold: float = 0.35
    refractory_seconds: float = 2.2
    ack_text: str = "Yes?"


@dataclass(frozen=True)
class WakeWordConfig:
    phrase: str
    model_name: str
    positives: tuple[PhraseSpec, ...]
    negatives: tuple[PhraseSpec, ...]
    output_dir: Path
    synthesis: SynthesisSettings = field(default_factory=SynthesisSettings)
    background: BackgroundSettings = field(default_factory=BackgroundSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    listen: ListenSettings = field(default_factory=ListenSettings)

    @property
    def clips_dir(self) -> Path:
        return self.output_dir / "clips"

    @property
    def enroll_dir(self) -> Path:
        return self.output_dir / "enroll"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{self.model_name}.onnx"


_MODEL_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _phrases(raw: Any, section: str) -> tuple[PhraseSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"[[{section}]] needs at least one entry")
    specs = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            raise ConfigError(f"[[{section}]] entry {i} needs a non-empty 'text'")
        clips = int(item.get("clips", 10))
        weight = int(item.get("weight", 1))
        if clips < 1 or weight < 0:
            raise ConfigError(f"[[{section}]] '{item['text']}': clips must be >= 1 and weight >= 0")
        specs.append(PhraseSpec(text=str(item["text"]).strip(), clips=clips, weight=weight))
    return tuple(specs)


def _section(cls, raw: dict[str, Any] | None, *, path_fields=(), tuple_fields=()):
    raw = dict(raw or {})
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown {cls.__name__} keys: {', '.join(sorted(unknown))}")
    for key in path_fields:
        if key in raw:
            value = raw[key]
            raw[key] = tuple(Path(v) for v in value) if isinstance(value, list) else Path(value)
    for key in tuple_fields:
        if key in raw:
            raw[key] = tuple(raw[key])
    return cls(**raw)


def config_from_dict(data: dict[str, Any]) -> WakeWordConfig:
    ww = data.get("wakeword") or {}
    phrase = str(ww.get("phrase", "")).strip()
    if not phrase:
        raise ConfigError("[wakeword] phrase is required")
    model_name = str(ww.get("model_name") or slugify(phrase))
    if not _MODEL_NAME_RE.match(model_name):
        raise ConfigError("[wakeword] model_name may only contain a-z, 0-9 and _")

    synthesis = _section(
        SynthesisSettings,
        data.get("synthesis"),
        path_fields=("voices_dir",),
        tuple_fields=("voices", "length_scales", "noise_scales", "noise_w_scales"),
    )
    if not synthesis.voices:
        raise ConfigError("[synthesis] voices must list at least one Piper voice")

    training = _section(TrainingSettings, data.get("training"))
    if not 0.0 <= training.holdout_fraction < 1.0:
        raise ConfigError("[training] holdout_fraction must be in [0, 1)")

    listen = _section(ListenSettings, data.get("listen"))
    if not 0.0 < listen.threshold < 1.0:
        raise ConfigError("[listen] threshold must be between 0 and 1")

    return WakeWordConfig(
        phrase=phrase,
        model_name=model_name,
        positives=_phrases(data.get("positive"), "positive"),
        negatives=_phrases(data.get("negative"), "negative"),
        output_dir=Path(ww.get("output_dir", f"runs/{model_name}")),
        synthesis=synthesis,
        background=_section(BackgroundSettings, data.get("background"), path_fields=("dirs",)),
        training=training,
        listen=listen,
    )


def load_config(path: str | Path) -> WakeWordConfig:
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return config_from_dict(data)


def slugify(text: str) -> str:
    """'Hey, Joy!' -> 'hey_joy'."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "phrase"
