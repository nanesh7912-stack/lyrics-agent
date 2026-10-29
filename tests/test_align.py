import numpy as np
import pytest
from ai_lyrics_hero.align import align_from_lrc, align_with_asr, align_estimated, voiced_regions, align_lyrics
from ai_lyrics_hero.asr import AsrWord
from ai_lyrics_hero.lyrics import parse_lyrics
from ai_lyrics_hero.ass_writer import write_ass
from ai_lyrics_hero.ass_parser import parse_ass
from ai_lyrics_hero.audio import SR


def test_lrc_alignment_exact_starts_and_no_overlap():
    lines, _ = parse_lyrics("[00:01.00]one two three\n[00:05.00]four five")
    ev, rp = align_from_lrc(lines)
    assert ev[0].start == 1.0 and ev[1].start == 5.0 and ev[0].end <= 5.0
    assert all(a.end <= b.start + 1e-6 for e in ev for a, b in zip(e.words, e.words[1:]))


def test_asr_alignment_keeps_user_spelling_and_uses_asr_times():
    lines, _ = parse_lyrics("Teri aankhen hain jaadu\nDil bar dil bar")
    asr = [AsrWord("teri", 1.0, 1.3), AsrWord("ankhein", 1.3, 1.8), AsrWord("hain", 1.8, 2.0), AsrWord("jadu", 2.0, 2.6),
           AsrWord("dilbar", 4.0, 4.6), AsrWord("dilbar", 4.8, 5.4)]
    ev, rp = align_with_asr(lines, asr, 10.0)
    assert [w.text for w in ev[0].words] == ["Teri", "aankhen", "hain", "jaadu"]       # spelling untouched
    assert ev[0].words[1].start == 1.3 and ev[0].timing_reliable
    assert ev[1].start >= 4.0 and rp[0].method == "asr"


def test_asr_missing_words_are_interpolated_and_flagged():
    lines, _ = parse_lyrics("alpha beta gamma delta")
    asr = [AsrWord("alpha", 1.0, 1.4), AsrWord("delta", 3.0, 3.4)]
    ev, rp = align_with_asr(lines, asr, 10.0)
    w = ev[0].words
    assert 1.4 <= w[1].start < w[2].start < w[3].start == 3.0
    assert not ev[0].timing_reliable and any("no match" in n for n in rp[0].notes)


def test_repeated_chorus_aligns_in_order():
    lines, _ = parse_lyrics("la la hey\nla la hey")
    asr = [AsrWord(t, 1 + i * 0.5, 1.4 + i * 0.5) for i, t in enumerate("la la hey la la hey".split())]
    ev, _ = align_with_asr(lines, asr, 10.0)
    assert ev[0].end <= ev[1].start + 1e-6 and ev[1].words[0].start == 2.5


def test_completely_wrong_transcript_raises_clear_error():
    lines, _ = parse_lyrics("hello world")
    with pytest.raises(RuntimeError):
        align_with_asr(lines, [AsrWord("zzzzqq", 1, 2)], 5.0)



def test_asr_timestamps_beyond_audio_duration_are_bounded():
    lines, _ = parse_lyrics("hello world")
    asr = [
        AsrWord("hello", 1.0, 1.3),
        AsrWord("world", 50.0, 50.4),
    ]

    events, reports = align_with_asr(lines, asr, 5.0)

    for event in events:
        for word in event.words:
            assert 0.0 <= word.start < 5.0
            assert word.start < word.end <= 5.0


def test_invalid_asr_timestamps_do_not_create_invalid_word_times():
    lines, _ = parse_lyrics("alpha beta gamma")
    asr = [
        AsrWord("alpha", -2.0, -1.0),
        AsrWord("beta", 1.0, 1.0),
        AsrWord("gamma", 2.0, 2.4),
    ]

    events, reports = align_with_asr(lines, asr, 5.0)

    for event in events:
        for word in event.words:
            assert 0.0 <= word.start < 5.0
            assert word.start < word.end <= 5.0

def _speech_like(sr=SR, segs=((1.0, 3.0), (4.0, 6.0))):
    x = np.zeros(sr * 8, np.float32)
    for a, b in segs:
        t = np.arange(int((b - a) * sr)) / sr
        x[int(a * sr):int(b * sr)] = 0.5 * np.sin(2 * np.pi * 200 * t)
    return x


def test_voiced_regions_found():
    regs, _, _ = voiced_regions(_speech_like())
    assert len(regs) == 2 and abs(regs[0][0] - 1.0) < 0.3 and abs(regs[1][1] - 6.0) < 0.3


def test_estimated_alignment_uses_voiced_time_and_is_flagged():
    lines, _ = parse_lyrics("one two three four\nfive six seven eight")
    ev, rp = align_estimated(lines, _speech_like(), 8.0)
    assert abs(ev[0].start - 1.0) < 0.4
    for e in ev:                                      # words are placed only where sound actually is
        for w in e.words:
            assert 0.9 <= w.start <= 3.1 or 3.9 <= w.start <= 6.1, w
    assert all(r.method == "estimated" and r.needs_review for r in rp)


def test_vocal_window_restricts_estimate():
    x = _speech_like(segs=((0.5, 1.0), (4.0, 6.0)))
    lines, _ = parse_lyrics("only line here")
    ev, _ = align_estimated(lines, x, 8.0, window=(3.5, 6.5))
    assert ev[0].start >= 3.9


def test_estimate_without_audio_still_returns_events():
    lines, _ = parse_lyrics("a b c")
    ev, _ = align_estimated(lines, None, 12.0)
    assert ev[0].start == 0.0


def test_ass_roundtrip_preserves_text_and_timing(tmp_path):
    lines, _ = parse_lyrics("[00:01.00]Dil bar jaani\n[00:04.00]मैंने ना ये जाना")
    ev, _ = align_from_lrc(lines)
    p = write_ass(ev, str(tmp_path / "x.ass"))
    back = parse_ass(p)
    assert [e.text for e in back] == [e.text for e in ev]
    assert abs(back[1].words[2].start - ev[1].words[2].start) < 0.02



def test_unreasonably_long_asr_word_does_not_stretch_lyric_line():
    lines, _ = parse_lyrics("aa la la")

    asr = [
        AsrWord("aa", 1.0, 12.0),
        AsrWord("la", 12.1, 12.4),
        AsrWord("la", 12.5, 12.8),
    ]

    events, reports = align_with_asr(lines, asr, 15.0)

    assert events
    assert events[0].end - events[0].start < 6.0, (
        "A single suspicious ASR word must not stretch the lyric line "
        "across an implausibly long interval."
    )


def test_asr_alignment_does_not_create_overlapping_lyric_lines():
    lines, _ = parse_lyrics("la la hey\nla la hey")

    asr = [
        AsrWord("la", 1.0, 1.3),
        AsrWord("la", 1.4, 1.7),
        AsrWord("hey", 1.8, 2.2),
        AsrWord("la", 5.0, 5.3),
        AsrWord("la", 5.4, 5.7),
        AsrWord("hey", 5.8, 6.2),
    ]

    events, reports = align_with_asr(lines, asr, 10.0)

    assert len(events) == 2
    assert events[0].end <= events[1].start + 1e-6, (
        "Chronologically ordered lyric lines must not overlap."
    )


def test_final_asr_event_does_not_exceed_audio_duration():
    lines, _ = parse_lyrics("alpha beta")
    asr = [
        AsrWord("alpha", 3.0, 3.3),
        AsrWord("beta", 4.0, 4.3),
    ]

    events, _ = align_with_asr(lines, asr, 5.0)

    assert events
    assert all(0.0 <= event.start < 5.0 for event in events)
    assert all(event.end <= 5.0 for event in events)



def test_karaoke_word_durations_remain_plausible():
    """ASR alignment must not assign multi-second durations to short vocalizations."""
    lines, _ = parse_lyrics("la la ... la la la")

    asr = [
        AsrWord("la", 10.0, 10.3),
        AsrWord("la", 10.3, 10.6),
        AsrWord("la", 10.7, 11.0),
        AsrWord("la", 11.1, 11.4),
    ]

    events, _ = align_with_asr(lines, asr, 15.0)

    assert events, "Alignment should produce at least one lyric event"

    for event in events:
        for word in event.words:
            assert 0 < word.end - word.start <= 3.5, (
                f"Implausible karaoke duration for {word.text!r}: "
                f"{word.start:.2f}-{word.end:.2f}"
            )


def test_short_vocalization_line_does_not_stretch_to_audio_end():
    """Short repeated vocalizations must not inherit a long trailing interval."""
    lines, _ = parse_lyrics("la la la ... la la la")

    asr = [
        AsrWord("la", 306.0, 306.3),
        AsrWord("la", 306.3, 306.6),
        AsrWord("la", 306.7, 307.0),
        AsrWord("la", 307.0, 307.3),
        AsrWord("la", 307.4, 307.7),
    ]

    events, _ = align_with_asr(lines, asr, 334.274)

    assert events, "Expected an alignment event"

    for event in events:
        assert event.end - event.start < 6.0, (
            f"Vocalization line unexpectedly spans "
            f"{event.end - event.start:.2f} seconds"
        )

        for word in event.words:
            assert word.end - word.start <= 3.5, (
                f"Unreasonable duration for {word.text!r}: "
                f"{word.end - word.start:.2f} seconds"
            )


def test_low_confidence_distant_vocalization_anchor_is_rejected():
    """A weak, distant ASR match must not stretch a short vocalization."""
    lines, _ = parse_lyrics("la la la ... la la la")

    asr = [
        AsrWord("la", 306.0, 306.3, prob=0.91),
        AsrWord("la", 306.3, 306.6, prob=0.88),
        AsrWord("la", 306.7, 307.0, prob=0.86),
        AsrWord("la", 307.0, 307.3, prob=0.91),
        AsrWord("la", 307.4, 307.7, prob=0.92),
        AsrWord("la,", 332.84, 334.24, prob=0.298),
    ]

    events, reports = align_with_asr(lines, asr, 334.274)

    assert events, "Expected an alignment event"
    assert events[0].end - events[0].start < 6.0, (
        "A low-confidence, distant ASR match stretched the vocalization "
        f"to {events[0].end - events[0].start:.2f} seconds."
    )

    assert all(word.end - word.start <= 3.5 for event in events for word in event.words), (
        "A vocalization word has an implausibly long duration."
    )


def test_repeated_vocalization_lines_stay_in_order_when_asr_words_are_sparse():
    """Sparse ASR matches must not move repeated lyric lines before their predecessor."""
    vocalization = "La la la ... la la la"
    lyrics = "Ban jaaye afsana\n" + "\n".join([vocalization] * 4)
    lines, _ = parse_lyrics(lyrics)

    asr = [
        AsrWord("ban", 302.58, 303.00, prob=0.90),
        AsrWord("jaaye", 303.00, 303.86, prob=0.85),
        AsrWord("afsana", 303.86, 305.52, prob=0.90),
        AsrWord("la", 305.52, 306.00, prob=0.90),
        AsrWord("la", 306.00, 306.30, prob=0.88),
        AsrWord("la", 306.30, 306.60, prob=0.86),
        AsrWord("la", 306.60, 306.90, prob=0.91),
        AsrWord("la", 306.90, 307.20, prob=0.92),
        AsrWord("la,", 332.84, 334.24, prob=0.298),
    ]

    events, reports = align_with_asr(lines, asr, 334.274)

    assert len(events) == 5, f"Expected five lyric events, got {len(events)}"

    assert events[-1].end >= 330.0, (
        "The final repeated vocalization block was compressed too early; "
        f"its end is {events[-1].end:.3f}s despite a matching low-confidence "
        "ASR token near the audio end."
    )

    for previous, current in zip(events, events[1:]):
        assert previous.start <= current.start + 1e-6, (
            f"Events are out of order: {previous.start:.3f}s then "
            f"{current.start:.3f}s"
        )
        assert previous.end <= current.start + 1e-6, (
            f"Lyric events overlap: {previous.start:.3f}-{previous.end:.3f}s "
            f"and {current.start:.3f}-{current.end:.3f}s"
        )


def test_high_confidence_false_asr_word_does_not_distort_lyric_timing():
    lines, _ = parse_lyrics("alpha beta gamma delta")
    asr = [
        AsrWord("alpha", 1.0, 1.3, 0.95),
        AsrWord("wrongword", 1.4, 1.7, 0.99),
        AsrWord("beta", 2.0, 2.3, 0.95),
        AsrWord("gamma", 2.5, 2.8, 0.95),
        AsrWord("delta", 3.0, 3.3, 0.95),
    ]

    events, reports = align_with_asr(lines, asr, 5.0)

    words = events[0].words
    assert [word.text for word in words] == ["alpha", "beta", "gamma", "delta"]
    assert words[0].start == pytest.approx(1.0)
    assert words[-1].start == pytest.approx(3.0)
    assert all(word.start < word.end for word in words)
