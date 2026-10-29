import pytest
from ai_lyrics_hero.align import align_from_lrc, align_with_asr
from ai_lyrics_hero.asr import AsrWord
from ai_lyrics_hero.ass_parser import parse_ass
from ai_lyrics_hero.ass_writer import write_ass
from ai_lyrics_hero.config import HeroConfig
from ai_lyrics_hero.layout import display_text
from ai_lyrics_hero.lyrics import parse_lyrics
from ai_lyrics_hero.text_utils import similarity, deva_to_roman, skeleton

LYRICS = """Tu Chale Aaye Aaye Aaye Aaye
Woh Jaan Jaaye Aaye Aaye Aaye Aaye
Gaalon Ki Khushboo Teri Aaye
I Feel So High High High High High
Aankhon Ko Palkon Ke Dhaagon Se Baandho Na
Neendein Uda Ke Kehte Ho Ke Jaago Na"""
LINES = LYRICS.split("\n")


def lrc(lines):
    return "\n".join(f"[00:{i * 4:02d}.00]{t}" for i, t in enumerate(lines))


def test_exact_text_survives_parse_align_write_and_reparse(tmp_path):
    lines, timed = parse_lyrics(lrc(LINES))
    ev, _ = align_from_lrc(lines)
    p = write_ass(ev, str(tmp_path / "x.ass"))
    back = parse_ass(p)
    assert [e.text for e in back] == LINES                                   # identical, incl. case and repeats
    assert [w.text for w in back[3].words] == "I Feel So High High High High High".split()


def test_punctuation_capitalization_ad_libs_and_mixed_language_are_untouched(tmp_path):
    src = ["Tu chale, aaye... (aaye!)", "I feel so HIGH - oh oh oh", "Dil ne kaha: \"ruk jaa\", Baby?", "Aaa-aaa  hmm"]
    lines, _ = parse_lyrics(lrc(src))
    ev, _ = align_from_lrc(lines)
    back = parse_ass(write_ass(ev, str(tmp_path / "p.ass")))
    assert [e.text for e in back] == ["Tu chale, aaye... (aaye!)", "I feel so HIGH - oh oh oh",
                                      "Dil ne kaha: \"ruk jaa\", Baby?", "Aaa-aaa hmm"]   # only whitespace runs collapse


def test_text_is_displayed_as_written_not_uppercased_by_default():
    assert display_text("Tu Chale Aaye", HeroConfig()) == "Tu Chale Aaye"
    assert display_text("Tu Chale Aaye", HeroConfig(uppercase=True)) == "TU CHALE AAYE"


def test_devanagari_is_not_substituted_for_romanized_lyrics_after_asr_alignment():
    lines, _ = parse_lyrics("Aankhon Ko Palkon Ke Dhaagon Se Baandho Na")
    asr = [AsrWord(t, 1 + i * 0.4, 1.35 + i * 0.4) for i, t in enumerate("आँखों को पलकों के धागों से बाँधो ना".split())]
    ev, rp = align_with_asr(lines, asr, 20.0)
    assert [w.text for w in ev[0].words] == "Aankhon Ko Palkon Ke Dhaagon Se Baandho Na".split()   # still Latin, exact
    assert all(w["source"] == "asr" and w["confidence"] >= 0.9 for w in rp[0].words)
    assert abs(ev[0].words[0].start - 1.0) < 1e-6 and ev[0].timing_reliable and not rp[0].uncertain_words


@pytest.mark.parametrize("roman, deva", [
    ("Aankhon", "आँखों"), ("Palkon", "पलकों"), ("Dhaagon", "धागों"), ("Baandho", "बाँधो"), ("Neendein", "नींदें"),
    ("Uda", "उड़ा"), ("Kehte", "कहते"), ("Jaago", "जागो"), ("Aaye", "आए"), ("Aaye", "आये"), ("Jaaye", "जाए"),
    ("Jaan", "जान"), ("Gaalon", "गालों"), ("Khushboo", "खुशबू"), ("Chale", "चले"), ("Tu", "तू"), ("Woh", "वो"),
    ("Ki", "की"), ("Teri", "तेरी"), ("High", "हाई"), ("Feel", "फील"), ("Zindagi", "ज़िंदगी"),
])
def test_romanized_and_devanagari_forms_match(roman, deva):
    assert similarity(roman, deva) >= 0.9, (roman, deva, deva_to_roman(deva))


def test_unrelated_words_do_not_match_across_scripts():
    for a, b in [("Aankhon", "रात"), ("Dil", "गाड़ी"), ("High", "तेरी"), ("Neendein", "जान")]:
        assert similarity(a, b) < 0.55, (a, b)


def test_full_hindi_song_with_devanagari_asr_aligns_every_line_in_order():
    lines, _ = parse_lyrics(LYRICS)
    deva = ("तू चले आए आए आए आए वो जान जाए आए आए आए आए गालों की खुशबू तेरी आए आई फील सो हाई हाई हाई हाई हाई "
            "आँखों को पलकों के धागों से बाँधो ना नींदें उड़ा के कहते हो के जागो ना").split()
    asr = [AsrWord(t, 1.0 + i * 0.45, 1.4 + i * 0.45) for i, t in enumerate(deva)]
    ev, rp = align_with_asr(lines, asr, 60.0)
    assert [e.text for e in ev] == LINES
    assert all(a.end <= b.start + 1e-6 for a, b in zip(ev, ev[1:]))                    # strictly ordered, no overlaps
    assert sum(len(r.uncertain_words) for r in rp) <= 3                                # nearly everything anchored


def test_missing_asr_words_are_flagged_interpolated_never_invented_or_called_exact():
    lines, _ = parse_lyrics("Gaalon Ki Khushboo Teri Aaye")
    asr = [AsrWord("गालों", 1.0, 1.4), AsrWord("आए", 3.0, 3.4)]                         # three words never recognized
    ev, rp = align_with_asr(lines, asr, 20.0)
    assert [w.text for w in ev[0].words] == "Gaalon Ki Khushboo Teri Aaye".split()
    assert not ev[0].timing_reliable and rp[0].needs_review
    srcs = [w["source"] for w in rp[0].words]
    assert srcs == ["asr", "interpolated", "interpolated", "interpolated", "asr"]
    assert [u["text"] for u in rp[0].uncertain_words] == ["Ki", "Khushboo", "Teri"]
    assert 1.4 <= ev[0].words[1].start < ev[0].words[2].start < ev[0].words[3].start <= 3.0


def test_split_and_merged_tokens_across_scripts():
    lines, _ = parse_lyrics("Dil bar jaane jaan")
    asr = [AsrWord("दिलबर", 1.0, 1.6), AsrWord("जान", 1.7, 2.0), AsrWord("ए", 2.0, 2.2), AsrWord("जान", 2.3, 2.7)]
    ev, rp = align_with_asr(lines, asr, 10.0)
    assert [w.text for w in ev[0].words] == ["Dil", "bar", "jaane", "jaan"]
    assert ev[0].words[0].start == 1.0 and ev[0].words[1].start >= 1.0


def test_wrong_language_transcript_raises_instead_of_inventing_timing():
    lines, _ = parse_lyrics("Aankhon Ko Palkon")
    with pytest.raises(RuntimeError):
        align_with_asr(lines, [AsrWord("completely", 1, 2), AsrWord("different", 2, 3)], 10.0)


def test_skeleton_is_only_a_matching_aid_and_does_not_change_inputs():
    w = "Aankhon"
    skeleton(w)
    assert w == "Aankhon" and skeleton("Aaye") == skeleton("आए")

def test_cross_script_similarity_rejects_consonant_skeleton_false_positive():
    # baithe should not match the different word baat.
    assert similarity("baithe", "बात") < 0.55

    # The correct Devanagari spelling must still match.
    assert similarity("baithe", "बैठे") >= 0.9
