import numpy as np
import pytest

from wakeword_lab.cli import build_parser
from wakeword_lab.devices import score_device
from wakeword_lab.listen import TriggerGate, meter
from wakeword_lab.train import window_metrics


def test_gate_fires_once_per_utterance():
    gate = TriggerGate(threshold=0.6, refractory=1.0)
    stream = [(0.0, 0.1), (0.1, 0.8), (0.2, 0.9), (0.5, 0.85), (1.2, 0.8), (1.3, 0.1), (1.4, 0.9)]
    fired = [t for t, s in stream if gate.update(s, t)]
    # 0.1 fires; 1.2 is past refractory but the score never dropped, so it waits to re-arm.
    assert fired == [0.1, 1.4]


def test_meter():
    assert meter(0.0, 4) == "----"
    assert meter(0.5, 4) == "##--"
    assert meter(2.0, 4) == "####"


def test_device_scoring_prefers_real_mics():
    assert score_device("USB Microphone", "Windows WASAPI") > score_device("USB Microphone", "MME")
    assert score_device("Steam Streaming Microphone", "Windows WASAPI") < 0
    assert score_device("Microsoft Sound Mapper - Input", "MME") < 0
    assert score_device("MacBook Pro Microphone", "Core Audio") > 0


def test_window_metrics():
    m = window_metrics(np.array([0.9, 0.2, 0.7, 0.1]), np.array([1, 1, 0, 0]))
    assert m == {"accuracy": 0.5, "recall": 0.5, "false_positive_rate": 0.5}


@pytest.mark.parametrize("cmd", ["synth", "train", "evaluate", "listen", "demo", "devices", "negatives"])
def test_cli_parses_every_command(cmd):
    args = build_parser().parse_args([cmd])
    assert callable(args.func)


def test_cli_evaluate_flags():
    args = build_parser().parse_args(["evaluate", "-c", "x.toml", "--threshold", "0.7", "--all"])
    assert args.config == "x.toml" and args.threshold == 0.7 and args.all
