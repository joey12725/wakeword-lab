import numpy as np
import pytest

from wakeword_lab.metrics import (
    count_activations,
    false_accepts_per_hour,
    format_table,
    overall,
    sliding_windows,
    summarize_peaks,
)


def test_count_activations_respects_refractory():
    scores = np.array([0, 0.9, 0.95, 0.9, 0, 0, 0.8, 0, 0.99])
    assert count_activations(scores, 0.5) == 5
    assert count_activations(scores, 0.5, refractory_frames=1) == 4
    assert count_activations(scores, 0.5, refractory_frames=2) == 2
    assert count_activations(scores, 0.5, refractory_frames=10) == 1
    assert count_activations(scores, 1.0) == 0


def test_false_accepts_per_hour():
    one_hour = int(3600 / 0.08)
    assert false_accepts_per_hour(3, one_hour) == pytest.approx(3.0)
    assert false_accepts_per_hour(3, 0) == 0.0


def test_sliding_windows_shape_and_content():
    feats = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    win = sliding_windows(feats, 16)
    assert win.shape == (5, 16, 3)
    np.testing.assert_array_equal(win[2], feats[2:18])
    assert sliding_windows(win, 16) is win  # already windowed
    with pytest.raises(ValueError):
        sliding_windows(feats[:4], 16)


def test_summarize_orders_positives_then_worst_confusables():
    rows = [
        ("Hey Joy.", 1, 0.9),
        ("Hey Joy.", 1, 0.4),
        ("Enjoy.", 0, 0.1),
        ("Hey Joey.", 0, 0.7),
        ("Hey Joey.", 0, 0.2),
    ]
    results = summarize_peaks(rows, threshold=0.5)
    assert [r.text for r in results] == ["Hey Joy.", "Hey Joey.", "Enjoy."]
    assert results[0].trigger_rate == 0.5
    assert results[1].triggered == 1

    summary = overall(results)
    assert summary["recall"] == 0.5
    assert summary["false_accept_rate"] == pytest.approx(1 / 3)
    assert "Hey Joey." in format_table(results, 0.5)
