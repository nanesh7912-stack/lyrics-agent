from ai_lyrics_hero.align import align_with_asr
from ai_lyrics_hero.asr import AsrWord
from ai_lyrics_hero.lyrics import parse_lyrics


def test_devanagari_asr_matches_romanized_hindi():
    lines, _ = parse_lyrics("kya hua")
    words = [AsrWord("क्या", 1.0, 1.3), AsrWord("हुआ", 1.3, 1.7)]
    events, reports = align_with_asr(lines, words, 5.0)
    assert events[0].words[0].start == 1.0
    assert events[0].words[1].start == 1.3
    assert reports[0].confidence > 0.7


def test_english_translation_inherits_hindi_line_window():
    lines, _ = parse_lyrics(
        "Are re are yeh kya hua\n"
        "Oh my, what has happened\n"
        "Maine na yeh jaana\n"
        "I did not know this"
    )
    words = [
        AsrWord("अरे", 1.0, 1.3), AsrWord("अरे", 1.3, 1.6),
        AsrWord("क्या", 1.6, 1.9), AsrWord("हुआ", 1.9, 2.3),
        AsrWord("मैंने", 4.0, 4.4), AsrWord("ना", 4.4, 4.6),
        AsrWord("ये", 4.6, 4.8), AsrWord("जाना", 4.8, 5.2),
    ]
    events, reports = align_with_asr(lines, words, 10.0)
    assert len(events) == 4
    assert (events[0].start, events[0].end) == (events[1].start, events[1].end)
    assert (events[2].start, events[2].end) == (events[3].start, events[3].end)
    assert "translation paired" in " ".join(reports[1].notes)


def test_vocalizations_are_not_misclassified_as_english():
    lines, _ = parse_lyrics("Aa aa ha ha tu ru ru ru\nLa la la la la")
    words = [
        AsrWord("अरे", 10.0, 10.4), AsrWord("अरे", 10.4, 10.8),
    ]
    # There are no obvious English translations; the algorithm must not pair
    # one vocalization line as a translation of the previous one.
    try:
        events, reports = align_with_asr(lines, words, 20.0)
    except RuntimeError:
        # Some models/texts may provide no match for these vocalizations; this
        # is a clear error, not a false translation pairing.
        return
    assert len(events) == 2
    assert not any("translation paired" in " ".join(r.notes) for r in reports)
