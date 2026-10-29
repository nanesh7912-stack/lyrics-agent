import pytest

from ai_lyrics_hero.text_utils import similarity


@pytest.mark.parametrize(
    "roman, devanagari",
    [
        ("Aankhon", "\u0906\u0901\u0916\u094b\u0902"),
        ("Palkon", "\u092a\u0932\u0915\u094b\u0902"),
        ("Dhaagon", "\u0927\u093e\u0917\u094b\u0902"),
        ("Baandho", "\u092c\u093e\u0901\u0927\u094b"),
        ("Neendein", "\u0928\u0940\u0902\u0926\u0947\u0902"),
        ("Uda", "\u0909\u0921\u093c\u093e"),
        ("Kehte", "\u0915\u0939\u0924\u0947"),
        ("Jaago", "\u091c\u093e\u0917\u094b"),
        ("Aaye", "\u0906\u090f"),
        ("Aaye", "\u0906\u092f\u0947"),
        ("Jaaye", "\u091c\u093e\u090f"),
        ("Jaan", "\u091c\u093e\u0928"),
        ("Gaalon", "\u0917\u093e\u0932\u094b\u0902"),
        ("Khushboo", "\u0916\u0941\u0936\u092c\u0942"),
        ("Chale", "\u091a\u0932\u0947"),
        ("Tu", "\u0924\u0942"),
        ("Woh", "\u0935\u094b"),
        ("Ki", "\u0915\u0940"),
        ("Teri", "\u0924\u0947\u0930\u0940"),
        ("High", "\u0939\u093e\u0908"),
        ("Feel", "\u092b\u0940\u0932"),
        ("Zindagi", "\u091c\u093c\u093f\u0902\u0926\u0917\u0940"),
        ("kuch", "\u0915\u0941\u091b"),
        ("baat", "\u092c\u093e\u0924"),
        ("maine", "\u092e\u0948\u0902\u0928\u0947"),
        ("baithe", "\u092c\u0948\u0920\u0947"),
        ("woh", "\u0935\u094b"),
        ("afsana", "\u0905\u092b\u093c\u0938\u093e\u0928\u093e"),
    ],
)
def test_valid_romanized_hindi_matches_devanagari(roman, devanagari):
    assert similarity(roman, devanagari) >= 0.90


@pytest.mark.parametrize(
    "roman, devanagari",
    [
        ("Aankhon", "\u0930\u093e\u0924"),
        ("Dil", "\u0917\u093e\u0921\u093c\u0940"),
        ("High", "\u0924\u0947\u0930\u0940"),
        ("Neendein", "\u091c\u093e\u0928"),
        ("baithe", "\u092c\u093e\u0924"),
        ("jaaye", "\u091c\u093e\u0913"),
        ("jaaye", "\u091c\u093e\u0939\u0940"),
        ("hum", "\u0924\u0941\u092e"),
        ("nikal", "\u091a\u0932\u0947"),
        ("na", "\u0928\u0939\u0940\u0902"),
    ],
)
def test_cross_script_false_matches_are_rejected(roman, devanagari):
    assert similarity(roman, devanagari) < 0.55


def test_unrelated_same_script_words_remain_dissimilar():
    assert similarity("car", "bike") < 0.50


def test_pehchana_romanization_matches_devanagari():
    assert similarity("pehchana", "\u092a\u0939\u091a\u093e\u0928\u093e") >= 0.90
    assert similarity("pehchaana", "\u092a\u0939\u091a\u093e\u0928\u093e") >= 0.90

