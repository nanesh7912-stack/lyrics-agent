from ai_lyrics_hero.animation import hero_state, hero_start_offset
from ai_lyrics_hero.config import HeroConfig
from ai_lyrics_hero.hero import select_heroes
from helpers import ev


def make(text="Maine na yeh jaana", **kw):
    cfg = HeroConfig(**kw)
    return select_heroes([ev(text, start=10.0, end=13.0)], cfg)[0], cfg


def test_hidden_before_event_and_after():
    e, c = make()
    assert not hero_state(9.9, e, c).active and not hero_state(13.0, e, c).active


def test_start_state_support_fades_hero_hidden_until_sung():
    e, c = make()
    s = hero_state(10.0, e, c)
    assert s.active and s.support_alpha == 0 and s.hero_alpha == 0
    off = hero_start_offset(e, c)
    assert off > 0                                    # not before the word is sung
    assert hero_state(10 + off - 0.01, e, c).hero_alpha == 0
    assert hero_state(10 + off + 0.02, e, c).hero_alpha > 0


def test_scale_overshoots_then_settles_to_one():
    e, c = make()
    t0 = 10 + hero_start_offset(e, c)
    scales = [hero_state(t0 + i * 0.01, e, c).hero_scale for i in range(0, 120)]
    assert scales[1] < 1.0 and max(scales) > 1.0 and max(scales) < 1.2   # controlled overshoot
    assert abs(hero_state(t0 + 0.5, e, c).hero_scale - 1.0) < 1e-3
    assert hero_state(t0 + 0.7, e, c).glow == 1.0


def test_zero_intensity_has_no_overshoot():
    e, c = make(animation_intensity=0.0)
    t0 = 10 + hero_start_offset(e, c)
    assert all(hero_state(t0 + i * 0.01, e, c).hero_scale <= 1.0 + 1e-9 for i in range(100))


def test_exit_fades_everything_smoothly():
    e, c = make(exit_s=0.3)
    a = [hero_state(13.0 - d, e, c) for d in (0.5, 0.3, 0.15, 0.01)]
    assert a[0].hero_alpha == 1.0
    assert a[1].hero_alpha >= a[2].hero_alpha >= a[3].hero_alpha
    assert a[3].hero_alpha < 0.05 and a[3].support_alpha < 0.05 and a[3].glow < 0.05


def test_hero_start_stays_inside_event_even_with_bad_timing():
    e, c = make("one two three four")
    e.words[-1].start = 99.0                          # corrupt timing
    off = hero_start_offset(e, c)
    assert 0 <= off <= e.duration - c.exit_s


def test_continuous_without_jumps_at_60fps():
    e, c = make()
    prev = None
    for i in range(int(3 * 60)):
        s = hero_state(10 + i / 60, e, c)
        if prev is not None:
            assert abs(s.hero_scale - prev.hero_scale) < 0.25 and abs(s.hero_alpha - prev.hero_alpha) < 0.5
        prev = s
