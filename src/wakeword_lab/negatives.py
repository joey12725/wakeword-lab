"""Download public background-audio corpora for negative training data.

These are the small music and sound-effect samples openWakeWord publishes
for training custom models (CC-licensed subsets of FMA and FSD50K, 16 kHz
WAV). Point ``[background] dirs`` in your config at the extracted folders.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

CORPORA = {
    "fma": "https://f002.backblazeb2.com/file/openwakeword-resources/data/fma_sample.zip",
    "fsd50k": "https://f002.backblazeb2.com/file/openwakeword-resources/data/fsd50k_sample.zip",
}

# Precomputed openWakeWord features (~11 h of speech, music and noise) used
# for the false-accepts-per-hour check in `wakeword-lab evaluate`.
VALIDATION_FEATURES_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
)


def _download(url: str, dest: Path, log=print) -> None:
    log(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)


def download_corpus(name: str, data_dir: Path, log=print) -> Path:
    target = data_dir / name
    if target.exists():
        log(f"  {name}: already at {target}")
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{name}.zip"
    _download(CORPORA[name], archive, log)
    log(f"  extracting {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    archive.unlink()
    return target


def download_validation_features(data_dir: Path, log=print) -> Path:
    dest = data_dir / "validation_set_features.npy"
    if not dest.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        _download(VALIDATION_FEATURES_URL, dest, log)
    return dest
