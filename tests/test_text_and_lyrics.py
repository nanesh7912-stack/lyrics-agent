from ai_lyrics_hero.text_utils import skeleton, similarity, syllables, normalize
from ai_lyrics_hero.lyrics import parse_lyrics


def test_romanized_spelling_variants_match():
    assert similarity("aankhen", "ankhein") >= 0.8
    assert similarity("pehchaana", "pehchana") == 1.0
    assert similarity("dil", "dill") == 1.0
    assert similarity("car", "bike") < 0.5
    assert similarity("jaana", "afsaana") < 0.6


def test_devanagari_skeleton_ignores_vowel_signs():
    assert similarity("जाना", "जना") > 0.8 and similarity("दिल", "रात") < 0.5


def test_syllables_never_zero():
    assert syllables("hmm") == 1 and syllables("afsaana") >= 3 and syllables("जाना") >= 1


def test_plain_lyrics_and_section_markers():
    lines, timed = parse_lyrics("[Chorus]\nBlue eyes\n\n  dil   bar \n")
    assert [l.text for l in lines] == ["Blue eyes", "dil bar"] and not timed


def test_lrc_line_and_multiple_timestamps():
    lines, timed = parse_lyrics("[ti:x]\n[00:12.50]Hello world\n[00:01.00][00:30.00]La la\n")
    assert timed and [l.start for l in lines] == [1.0, 12.5, 30.0]


def test_enhanced_lrc_word_tags():
    lines, timed = parse_lyrics("[00:10.00]<00:10.00>Hello <00:10.50>big <00:11.00>world")
    assert lines[0].text == "Hello big world" and lines[0].word_starts == [10.0, 10.5, 11.0]


def test_normalize_keeps_devanagari():
    assert normalize("आँखें!") == "आँखें"
