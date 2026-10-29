import numpy as np
from ai_lyrics_hero.config import HeroConfig
from ai_lyrics_hero.hero import select_heroes
from ai_lyrics_hero.renderer import FrameRenderer
from ai_lyrics_hero.audio import Levels
from helpers import ev

W, H = 540, 960   # half size keeps tests fast


def renderer(text="Maine na yeh jaana", cfg=None, **kw):
    cfg = cfg or HeroConfig()
    events = select_heroes([ev(text, start=0.0, end=3.0, **kw)], cfg)
    return FrameRenderer(cfg, events, W, H, Levels(None), title="TEST SONG"), cfg


def arr(img):
    return np.asarray(img, dtype=np.int16)


def test_frame_size_and_title_present():
    r, _ = renderer()
    f = r.render(2.7)
    assert f.size == (W, H)
    title_band = arr(f)[int(H * 0.05):int(H * 0.10)]
    assert title_band.max() > 200                      # white title text is drawn


def test_hero_word_is_large_bright_and_centered_horizontally():
    r, _ = renderer()
    a = arr(r.render(2.7))
    mid = a[int(H * 0.40):int(H * 0.56)]
    bright = (mid.min(axis=2) > 235)
    assert bright.sum() > 2500
    cols = np.where(bright.any(axis=0))[0]
    assert abs((cols.min() + cols.max()) / 2 - W / 2) < W * 0.08
    assert cols.min() > W * 0.04 and cols.max() < W * 0.96         # no clipping at the frame edges


def test_glow_is_behind_the_word_and_localized():
    r, _ = renderer()
    on = arr(r.render(2.7)); off = arr(r.render(0.01))
    near = on[int(H * 0.47):int(H * 0.53), :].astype(float).mean()
    far = on[int(H * 0.70):int(H * 0.76), :].astype(float).mean()
    far_off = off[int(H * 0.70):int(H * 0.76), :].astype(float).mean()
    assert near > far
    assert abs(far - far_off) < 8                       # background far from the word is not brightened


def test_glow_intensity_zero_removes_halo():
    r1, _ = renderer(cfg=HeroConfig(glow_intensity=0.0))
    r2, _ = renderer(cfg=HeroConfig(glow_intensity=1.0))
    ring = (slice(int(H * 0.535), int(H * 0.58)), slice(0, W))
    assert arr(r2.render(2.7))[ring].mean() > arr(r1.render(2.7))[ring].mean()


def test_disabled_uses_plain_presentation_and_still_renders():
    r, _ = renderer(cfg=HeroConfig(enabled=False))
    a = arr(r.render(1.5))
    assert a[int(H * 0.40):int(H * 0.56)].max() > 200


def test_no_reliable_timing_still_renders_hero():
    r, _ = renderer("koi na pehchaana", reliable=False)
    assert arr(r.render(2.7))[int(H * 0.35):int(H * 0.60)].max() > 200


def test_devanagari_renders_non_empty():
    r, _ = renderer("मैंने ना ये जाना")
    a = arr(r.render(2.7))
    assert (a[int(H * 0.35):int(H * 0.6)].min(axis=2) > 210).sum() > 800


def test_gap_between_events_shows_background_only_no_text():
    cfg = HeroConfig()
    events = select_heroes([ev("hello world", 0, 2), ev("goodbye friend", 4, 6)], cfg)
    r = FrameRenderer(cfg, events, W, H, None, title="")
    mid = arr(r.render(3.0))[int(H * 0.3):int(H * 0.65)]
    assert (mid.min(axis=2) > 200).sum() == 0


def test_visualizer_reacts_to_audio():
    sr = 16000
    loud = (np.sin(np.arange(sr * 2) * 0.05) * 0.9).astype(np.float32)
    cfg = HeroConfig()
    events = select_heroes([ev("hello world", 0, 2)], cfg)
    quiet_r = FrameRenderer(cfg, events, W, H, Levels(np.zeros(sr * 2, np.float32)), title="")
    loud_r = FrameRenderer(cfg, events, W, H, Levels(loud), title="")
    zone = (slice(int(H * 0.82), int(H * 0.93)), slice(0, W))
    blue = lambda f: (arr(f)[zone][..., 2] > 200).sum()
    assert blue(loud_r.render(0.5)) > blue(quiet_r.render(0.5))
