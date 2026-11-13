from collections import Counter

from wakeword_lab.synth import plan_clips, plan_phrase, read_manifest, write_manifest


def test_plan_counts_match_config(config):
    jobs = plan_clips(config)
    expected = sum(p.clips for p in config.positives) + sum(n.clips for n in config.negatives)
    assert len(jobs) == expected
    assert sum(j.label for j in jobs) == sum(p.clips for p in config.positives)
    assert len({j.path for j in jobs}) == len(jobs)


def test_plan_is_deterministic(config):
    assert plan_clips(config) == plan_clips(config)


def test_plan_spreads_voices_and_prosody(config):
    jobs = plan_phrase(config.positives[0], 1, 0, config, {})
    voices = Counter(j.voice for j in jobs)
    assert set(voices) == set(config.synthesis.voices)
    assert max(voices.values()) - min(voices.values()) <= len(jobs) // 2
    assert len({(j.voice, j.length_scale, j.noise_scale, j.noise_w_scale) for j in jobs}) == len(jobs)


def test_multi_speaker_voice_uses_many_speakers(config):
    voice = config.synthesis.voices[0]
    jobs = plan_phrase(config.negatives[0], 0, 0, config, {voice: 900})
    speakers = {j.speaker for j in jobs if j.voice == voice}
    assert len(speakers) > 10
    assert all(j.path.startswith("negative/00_hey_joey/") for j in jobs)
    assert all(j.weight == config.negatives[0].weight for j in jobs)


def test_manifest_round_trip(config, tmp_path):
    jobs = plan_clips(config, {config.synthesis.voices[0]: 4})
    write_manifest(tmp_path, jobs)
    assert read_manifest(tmp_path) == jobs


def test_missing_manifest_is_empty(tmp_path):
    assert read_manifest(tmp_path) == []
