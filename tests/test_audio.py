import numpy as np
import pytest

from wakeword_lab.audio import (
    CLIP_SAMPLES,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    FrameChunker,
    load_wav_16k,
    make_beep,
    pad_center,
    resample_to_16k,
    rms,
    to_int16,
    write_wav_16k,
)


def test_to_int16_clips_range():
    out = to_int16(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert out.dtype == np.int16
    assert out.tolist() == [-32767, -32767, 0, 32767, 32767]


def test_rms():
    assert rms(np.zeros(0)) == 0.0
    assert rms(np.ones(10)) == pytest.approx(1.0)


@pytest.mark.parametrize("n", [100, CLIP_SAMPLES, CLIP_SAMPLES + 501])
def test_pad_center_length(n):
    assert pad_center(np.ones(n)).size == CLIP_SAMPLES


def test_pad_center_keeps_middle():
    audio = np.arange(10, dtype=np.float32)
    assert pad_center(audio, 4).tolist() == [3, 4, 5, 6]


def test_beep_fades_in_and_out():
    beep = make_beep(seconds=0.1)
    assert beep.size == int(0.1 * SAMPLE_RATE)
    assert abs(beep[0]) < 1e-6 and abs(beep[-1]) < 0.01
    assert np.max(np.abs(beep)) <= 0.25 + 1e-6


def test_chunker_emits_fixed_frames_and_keeps_remainder():
    chunker = FrameChunker(SAMPLE_RATE)
    assert chunker.push(np.zeros(FRAME_SAMPLES - 1)) == []
    frames = chunker.push(np.arange(FRAME_SAMPLES * 2, dtype=np.float32))
    assert len(frames) == 2
    assert all(f.size == FRAME_SAMPLES for f in frames)
    joined = np.concatenate(frames)
    assert joined[FRAME_SAMPLES - 1] == 0.0  # first input sample lands where the remainder ended


def test_resample_48k():
    pytest.importorskip("scipy")
    out = resample_to_16k(np.zeros(48_000), 48_000)
    assert out.size == SAMPLE_RATE


def test_wav_round_trip(tmp_path):
    pytest.importorskip("scipy")
    audio = (make_beep(seconds=0.5) * 2).astype(np.float32)
    path = tmp_path / "x" / "beep.wav"
    write_wav_16k(path, audio)
    back = load_wav_16k(path)
    assert back.size == audio.size
    assert np.max(np.abs(back - audio)) < 1e-3
