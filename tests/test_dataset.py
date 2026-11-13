import numpy as np

from wakeword_lab.dataset import WindowSet, augment_clip, is_holdout, slice_long_audio
from wakeword_lab.features import active_window_indices


def test_holdout_split_is_stable_and_roughly_sized():
    names = [f"positive/00_hey_joy/{i:05d}.wav" for i in range(4000)]
    picked = [n for n in names if is_holdout(n, 0.15)]
    assert picked == [n for n in names if is_holdout(n, 0.15)]
    assert 0.12 < len(picked) / len(names) < 0.18
    assert not any(is_holdout(n, 0.0) for n in names)


def test_augment_keeps_shape_and_range():
    rng = np.random.default_rng(0)
    audio = np.full(1000, 0.9, dtype=np.float32)
    for _ in range(20):
        out = augment_clip(audio, rng)
        assert out.shape == audio.shape and out.dtype == np.float32
        assert np.all(np.abs(out) <= 1.0)


def test_slice_long_audio():
    assert len(slice_long_audio(np.zeros(100), hop=10, length=200)) == 1
    pieces = slice_long_audio(np.arange(100.0), hop=25, length=50)
    assert [p[0] for p in pieces] == [0, 25, 50]
    assert all(p.size == 50 for p in pieces)


def test_active_windows_skip_silence():
    frame_rms = np.array([0.0, 0.001, 0.2, 0.3, 0.08, 0.001])
    assert active_window_indices(frame_rms) == [2, 3, 4]
    assert active_window_indices(np.full(4, 0.001)) == [0]  # quiet clip: loudest frame only
    assert active_window_indices(np.array([])) == []


def test_window_set_arrays():
    ws = WindowSet()
    assert ws.arrays()[0].shape == (0, 16, 96)
    ws.add([np.zeros((16, 96))] * 3, 1)
    X, y = ws.arrays()
    assert X.shape == (3, 16, 96) and X.dtype == np.float32
    assert y.tolist() == [1, 1, 1]
