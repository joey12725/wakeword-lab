import pytest

from wakeword_lab.config import ConfigError, config_from_dict, load_config, slugify

from .conftest import ROOT


def minimal(**overrides):
    data = {
        "wakeword": {"phrase": "Hey Nova"},
        "positive": [{"text": "Hey Nova.", "clips": 3}],
        "negative": [{"text": "Hey Nora.", "clips": 2, "weight": 4}],
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize("name", ["hey_joy.toml", "hey_joy_large.toml"])
def test_shipped_configs_load(name):
    cfg = load_config(ROOT / "configs" / name)
    assert cfg.phrase == "Hey Joy"
    assert cfg.model_path.name == "hey_joy.onnx"
    assert cfg.positives and cfg.negatives


def test_defaults_fill_in():
    cfg = config_from_dict(minimal())
    assert cfg.model_name == "hey_nova"
    assert str(cfg.output_dir) == "runs/hey_nova"
    assert cfg.negatives[0].weight == 4
    assert cfg.positives[0].weight == 1
    assert 0 < cfg.listen.threshold < 1


@pytest.mark.parametrize(
    "data, message",
    [
        (minimal(wakeword={}), "phrase is required"),
        (minimal(positive=[]), "positive"),
        (minimal(negative=[{"text": "", "clips": 1}]), "text"),
        (minimal(negative=[{"text": "x", "clips": 0}]), "clips"),
        (minimal(wakeword={"phrase": "Hey", "model_name": "Bad Name"}), "model_name"),
        (minimal(training={"epochz": 3}), "unknown"),
        (minimal(listen={"threshold": 1.5}), "threshold"),
        (minimal(training={"holdout_fraction": 1.0}), "holdout"),
    ],
)
def test_bad_configs_raise(data, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_slugify():
    assert slugify("Hey, Joy!") == "hey_joy"
    assert slugify("What's up?") == "what_s_up"
    assert slugify("!!!") == "phrase"
