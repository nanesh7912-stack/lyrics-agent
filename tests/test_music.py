import numpy as np
from ai_lyrics_hero.music import analyze, MusicMap, SR


def clicks(bpm, dur, amp=0.9, sr=SR):
    x = np.zeros(int(dur * sr), np.float32)
    step = 60 / bpm
    t = 0.2
    while t < dur - 0.1:
        i = int(t * sr)
        n = int(0.03 * sr)
        x[i:i + n] += amp * np.hanning(n) * np.sin(2 * np.pi * 120 * np.arange(n) / sr)
        t += step
    return x


def test_fallback_for_missing_short_or_bad_audio_is_safe_and_deterministic():
    for bad in (None, np.zeros(100, np.float32), np.full(SR * 3, np.nan, np.float32)):
        m = analyze(bad)
        assert not m.reliable and m.level_at(5.0) == "mid" and m.nearest_impact(5, 1, 1) is None
        assert m.notes and m.energy_at(1.0) == 0.5


def test_tempo_and_beat_grid_of_a_clear_pulse():
    m = analyze(clicks(120, 12))
    assert m.reliable and m.tempo is not None and abs(m.tempo - 120) < 4
    assert len(m.beats) >= 20 and abs(np.diff(m.beats).mean() - 0.5) < 0.02
    near = [min(abs(b - o) for o, _ in m.onsets) for b in m.beats[2:-2]]
    assert np.median(near) < 0.08                                     # grid sits on the actual hits


def test_onsets_and_impacts_found_at_the_hits():
    m = analyze(clicks(90, 10))
    assert len(m.onsets) >= 10
    assert all(any(abs(t - o) < 0.08 for o, _ in m.onsets) for t in [0.2 + 0.667 * k for k in range(2, 12)])
    assert 1 <= len(m.impacts) <= len(m.onsets)


def test_quiet_loud_sections_are_distinguished_and_offset_is_applied():
    sr = SR
    quiet = 0.02 * np.sin(2 * np.pi * 200 * np.arange(6 * sr) / sr)
    loud = clicks(110, 6) + 0.4 * np.sin(2 * np.pi * 200 * np.arange(6 * sr) / sr)
    m = analyze(np.concatenate([quiet, loud]).astype(np.float32), offset=10.0)
    assert m.level_at(13.0) == "quiet" and m.level_at(19.0) == "high"
    assert m.energy_at(10.0 + 3) < m.energy_at(10.0 + 9)


def test_silence_gap_detection():
    x = np.concatenate([clicks(100, 4), np.zeros(SR * 3, np.float32), clicks(100, 4)])
    m = analyze(x)
    assert m.is_quiet_gap(4.5, 6.5) and not m.is_quiet_gap(0.5, 3.5)


def test_sparse_audio_without_pulse_reports_no_beat_grid():
    rng = np.random.default_rng(0)
    x = (0.02 * rng.standard_normal(SR * 8)).astype(np.float32)
    m = analyze(x)
    assert m.tempo is None and m.beats == []


def test_deterministic():
    x = clicks(100, 8)
    a, b = analyze(x), analyze(x)
    assert a.beats == b.beats and a.impacts == b.impacts and np.array_equal(a.energy, b.energy)


def test_nearest_impact_window():
    m = MusicMap(impacts=[(5.0, 0.9), (7.0, 0.6)], reliable=True)
    assert m.nearest_impact(5.1, before=0.2, after=0.0)[1] == 5.0
    assert m.nearest_impact(5.1, before=0.05, after=0.0) is None
    assert m.nearest_impact(6.9, before=0.0, after=0.2)[1] == 7.0
