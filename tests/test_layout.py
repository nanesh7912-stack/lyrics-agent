import pytest
from ai_lyrics_hero.config import HeroConfig
from ai_lyrics_hero.hero import select_heroes
from ai_lyrics_hero.layout import compute_layout, SAFE_X, TITLE_ZONE_BOTTOM, VIZ_ZONE_TOP
from helpers import ev

W, H = 1080, 1920


def lay(text, cfg=None, **kw):
    cfg = cfg or HeroConfig()
    e = select_heroes([ev(text, **kw)], cfg)[0]
    return e, compute_layout(e, cfg, W, H)


def assert_safe(l):
    for t in l.all_lines():
        x0, y0, x1, y1 = t.rect
        assert x0 >= W * SAFE_X - 2 and x1 <= W * (1 - SAFE_X) + 2, (t.text, t.rect)
        assert y0 >= H * TITLE_ZONE_BOTTOM and y1 <= H * VIZ_ZONE_TOP, (t.text, t.rect)


def test_support_above_hero_and_hero_larger():
    e, l = lay("Maine na yeh jaana")
    assert l.mode == "hero" and l.above and l.hero_lines
    assert l.hero_lines[0].size > 2 * l.above[0].size
    assert l.above[-1].rect[3] <= l.hero_lines[0].rect[1]          # no collision
    assert_safe(l)


def test_text_after_hero_goes_below():
    e, l = lay("burning [[bright]] tonight", explicit=(1, 1))
    assert l.above and l.below
    assert l.hero_lines[-1].rect[3] <= l.below[0].rect[1]
    assert_safe(l)


def test_long_word_is_shrunk_not_clipped():
    e, l = lay("so " + "Supercalifragilisticexpialidocious")
    assert l.hero_lines[0].size < HeroConfig().hero_size
    assert_safe(l)


def test_long_phrase_wraps_to_two_lines():
    cfg = HeroConfig(max_phrase_words=4)
    e, l = lay("we [[remember every single beautiful moment together]] now", cfg=cfg, explicit=(1, 6))
    assert len(l.hero_lines) == 2
    assert_safe(l)


def test_unicode_layouts_stay_in_bounds():
    for t in ["मैंने ना ये जाना", "अरे रे अरे ये क्या हुआ", "koi na pehchaana"]:
        e, l = lay(t)
        assert l.hero_lines, t
        assert_safe(l)


def test_plain_fallback_when_no_hero():
    e, l = lay("are re are na to hai")
    assert e.hero is None and l.mode == "plain" and l.hero_lines
    assert_safe(l)


@pytest.mark.parametrize("size", [(1080, 1920), (720, 1280), (1080, 1350)])
def test_other_portrait_sizes(size):
    cfg = HeroConfig()
    e = select_heroes([ev("Maine na yeh jaana")], cfg)[0]
    l = compute_layout(e, cfg, *size)
    for t in l.all_lines():
        assert 0 <= t.rect[0] and t.rect[2] <= size[0] and 0 <= t.rect[1] and t.rect[3] <= size[1]
