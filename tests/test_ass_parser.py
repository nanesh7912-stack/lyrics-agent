from ai_lyrics_hero.ass_parser import parse_ass, parse_text

HDR = "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"


def test_karaoke_word_timing():
    evs = parse_ass(HDR + "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\k50}Hello {\\k150}world\n")
    e = evs[0]
    assert e.timing_reliable
    assert [w.text for w in e.words] == ["Hello", "world"]
    assert abs(e.words[0].start - 1.0) < 1e-6 and abs(e.words[1].start - 1.5) < 1e-6


def test_no_karaoke_means_interpolated_and_flagged_unreliable():
    e = parse_ass(HDR + "Dialogue: 0,0:00:00.00,0:00:04.00,Default,,0,0,0,,one three five\n")[0]
    assert not e.timing_reliable
    assert e.words[0].start == 0 and abs(e.words[-1].end - 4.0) < 1e-6
    assert all(a.end <= b.start + 1e-9 for a, b in zip(e.words, e.words[1:]))


def test_overrun_karaoke_is_rejected_as_unreliable():
    e = parse_ass(HDR + "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\k900}a {\\k900}b\n")[0]
    assert not e.timing_reliable and e.words[-1].end <= 1.0 + 1e-6


def test_explicit_hero_marker_and_tags_stripped():
    e = parse_ass(HDR + "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,{\\an5\\pos(1,2)}we [[never ever]] stop\\Nnow\n")[0]
    assert [w.text for w in e.words] == ["we", "never", "ever", "stop", "now"]
    assert e.explicit_hero == (1, 2)


def test_unicode_devanagari_and_commas_in_text():
    e = parse_ass(HDR + "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,अरे रे, क्या हुआ\n")[0]
    assert [w.text for w in e.words] == ["अरे", "रे,", "क्या", "हुआ"]


def test_bad_lines_are_skipped_not_fatal():
    evs = parse_ass(HDR + "Dialogue: 0,bad,time,Default,,0,0,0,,x\nDialogue: 0,0:00:05.00,0:00:04.00,D,,0,0,0,,y\n"
                    "Dialogue: 0,0:00:01.00,0:00:02.00,D,,0,0,0,,ok\n")
    assert [e.text for e in evs] == ["ok"]
