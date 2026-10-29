from ai_lyrics_hero.text_utils import similarity


def test_common_romanized_hindi_spellings_match_devanagari():
    assert similarity("kuch", "\u0915\u0941\u091b") >= 0.9
    assert similarity("maine", "\u092e\u0948\u0902\u0928\u0947") >= 0.9
    assert similarity("baithe", "\u092c\u0948\u0920\u0947") >= 0.9
    assert similarity("woh", "\u0935\u094b") >= 0.9


def test_cross_script_matching_still_rejects_false_positive():
    assert similarity("baithe", "\u092c\u093e\u0924") < 0.55
    assert similarity("car", "bike") < 0.55


def test_common_hindi_words_remain_compatible():
    pairs = [
        ("baat", "\u092c\u093e\u0924"),
        ("hoti", "\u0939\u094b\u0924\u0940"),
        ("rahe", "\u0930\u0939\u0947"),
        ("koi", "\u0915\u094b\u0908"),
        ("na", "\u0928\u093e"),
        ("pehchana", "\u092a\u0939\u091a\u093e\u0928\u093e"),
        ("hum", "\u0939\u092e"),
        ("jaaye", "\u091c\u093e\u090f"),
        ("nikal", "\u0928\u093f\u0915\u0932"),
        ("dil", "\u0926\u093f\u0932"),
        ("kahin", "\u0915\u0939\u0940\u0902"),
        ("yakeen", "\u092f\u0915\u0940\u0928"),
        ("jaana", "\u091c\u093e\u0928\u093e"),
        ("afsana", "\u0905\u092b\u093c\u0938\u093e\u0928\u093e"),
    ]

    for roman, hindi in pairs:
        score = similarity(roman, hindi)
        assert score >= 0.9, (
            f"Unexpectedly low similarity for {roman!r} and {hindi!r}: "
            f"{score:.3f}"
        )
