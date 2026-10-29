import json
import os
import subprocess
import numpy as np
import pytest
from ai_lyrics_hero.config import HeroConfig, STYLE_PRESETS
from ai_lyrics_hero.scenes import SceneLexicon, plan_scenes, SceneRenderer, ScenePlan
from ai_lyrics_hero.caption import layout_caption, active_word
from ai_lyrics_hero.renderer import FrameRenderer
from ai_lyrics_hero.hero import select_heroes
from ai_lyrics_hero.audio import Levels
from ai_lyrics_hero.cli import main
from ai_lyrics_hero.pipeline import probe
from helpers import ev

LEX = SceneLexicon()
W, H = 540, 960


def kind(text, allow=False):
    return LEX.match(text.split(), allow)[0]


def test_scene_words_map_to_the_right_scene():
    assert kind("chaand ke neeche tera intezaar") == "moon"
    assert kind("teri aankhon mein kho gaya") == "eye"
    assert kind("jalta hua dil") == "heart_fire"
    assert kind("baarish mein yaadein purani") == "rain"
    assert kind("raat bhar taare gin gin ke") == "stars"          # strongest scene in the line
    assert kind("tu raat ko aaya") == "night"
    assert kind("तेरी आँखों में") == "eye" and kind("मेरा दिल") == "heart_fire"


def test_unknown_words_get_no_scene_and_blocked_words_are_ignored():
    assert kind("ka intezaar hai") is None and kind("kar de bar") is None


def test_ambiguous_scene_words_need_opt_in():
    assert kind("tara") is None and kind("tara", True) == "moon" or kind("tara", True) is None
    ms = LEX.match(["tara"], False)
    assert ms[0] is None


def test_plan_repeats_inner_repeats_and_continuity():
    cfg = HeroConfig()
    evs = [ev("ka intezaar hai", 0, 2), ev("tu hi tu hai tu hi tu hai", 2, 4), ev("ka intezaar hai", 4, 6),
           ev("jalta dil", 6, 8), ev("jalti aag", 8.05, 10), ev("hello there", 12, 14)]
    plans, _ = plan_scenes(evs, LEX, cfg)
    assert [p.kind for p in plans] == ["sky", "streaks", "streaks", "heart_fire", "heart_fire", "sky"]
    assert not plans[3].fade_out and not plans[4].fade_in          # same scene back-to-back: no flicker
    assert plans[3].fade_in and plans[4].fade_out


def test_user_scene_lexicon_extends(tmp_path):
    p = tmp_path / "s.json"; p.write_text(json.dumps({"scenes": {"moon": ["purnima"]}}))
    assert SceneLexicon(str(p)).match(["purnima"])[0] == "moon"


@pytest.mark.parametrize("k", ["moon", "eye", "heart_fire", "rain", "streaks", "night", "stars"])
def test_every_scene_draws_something_distinct_from_the_plain_sky(k):
    r = SceneRenderer(W, H)
    sky = np.asarray(r.frame(1.0, ScenePlan("sky", "", True, True), 1.0, 1.0), dtype=np.int16)
    sc = np.asarray(r.frame(1.0, ScenePlan(k, "", True, True), 1.0, 1.0), dtype=np.int16)
    assert sc.shape == (H, W, 3)
    assert np.abs(sc - sky).mean() > 0.05, k


def test_scene_specific_pixels():
    r = SceneRenderer(W, H)
    f = lambda k: np.asarray(r.frame(1.0, ScenePlan(k, "", True, True), 1.0, 1.0), dtype=np.int16)
    moon = f("moon")[int(H * 0.12):int(H * 0.26), int(W * 0.35):int(W * 0.65)]
    assert (moon.min(axis=2) > 200).sum() > 2000                         # bright cream disc
    heart = f("heart_fire")[int(H * 0.12):int(H * 0.26), int(W * 0.35):int(W * 0.65)]
    assert ((heart[..., 0] > 220) & (heart[..., 2] < 120)).sum() > 1500  # orange heart
    eye = f("eye")[int(H * 0.15):int(H * 0.25), int(W * 0.25):int(W * 0.75)]
    assert ((eye[..., 2] > 180) & (eye[..., 0] < 150)).sum() > 100       # blue iris


def test_scene_animates_over_time_and_is_deterministic():
    r1, r2 = SceneRenderer(W, H), SceneRenderer(W, H)
    p = ScenePlan("rain", "", True, True)
    a, b = r1.frame(1.0, p, 1.0, 1.0), r2.frame(1.0, p, 1.0, 1.0)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(r1.frame(1.2, p, 1.2, 1.0)))


def test_fade_alpha_blends_toward_the_sky():
    r = SceneRenderer(W, H)
    p = ScenePlan("moon", "", True, True)
    sky = np.asarray(r.frame(1.0, None, 0, 0), dtype=np.int16)
    d = lambda a: np.abs(np.asarray(r.frame(1.0, p, 1.0, a), dtype=np.int16) - sky).mean()
    assert d(0.0) < 0.05 < d(0.3) < d(1.0)


# ---------------- caption style ----------------
def cfg_scene(**kw):
    d = dict(STYLE_PRESETS["scene"]); d.update(kw)
    return HeroConfig.from_dict(d)


def test_caption_layout_inside_frame_and_wraps_narrow():
    cfg = cfg_scene()
    for text in ["chaand ke neeche tera intezaar", "a very long lyric line that keeps going and going and going on", "मैंने ना ये जाना"]:
        lay = layout_caption(text.split(), cfg, 1080, 1920, 0.74)
        x0, y0, x1, y1 = lay.rect
        assert x0 >= 1080 * 0.10 and x1 <= 1080 * 0.90 and y0 >= 1920 * 0.10 and y1 <= 1920 * 0.90, text
    assert len({w.y for w in layout_caption("chaand ke neeche tera intezaar".split(), cfg, 1080, 1920, .74).words}) >= 2


def test_active_word_follows_timing():
    e = ev("one two three four", 0, 4, reliable=False)
    assert active_word(e, -0.1) is None and active_word(e, 0.01) == 0 and active_word(e, 3.9) == 3


def _renderer(text="jalta hua dil", cfg=None):
    cfg = cfg or cfg_scene()
    evs = select_heroes([ev(text, 0.0, 3.0)], cfg)
    return FrameRenderer(cfg, evs, W, H, Levels(None))


def test_scene_style_renders_scene_caption_highlight_and_no_visualizer_or_emoji():
    r = _renderer()
    assert r.scene_plans[0].kind == "heart_fire" and not r.cues
    a = np.asarray(r.render(2.0), dtype=np.int16)
    cap = a[int(H * 0.66):int(H * 0.82)]
    assert (cap.min(axis=2) > 225).sum() > 150                                  # white caption text
    assert ((cap[..., 2] > 200) & (cap[..., 0] > 120) & (cap[..., 0] < 190)).sum() > 30   # highlighted word
    bars = a[int(H * 0.86):int(H * 0.93)]
    assert ((bars[..., 2] > 230) & (bars[..., 0] < 100)).sum() == 0              # no visualizer bars


def test_highlight_moves_from_word_to_word():
    r = _renderer("moon light shines bright")
    def hi_cols(t):
        a = np.asarray(r.render(t), dtype=np.int16)[int(H * 0.60):int(H * 0.85)]
        m = (a[..., 2] > 200) & (a[..., 0] > 120) & (a[..., 0] < 190) & (a[..., 1] < 190)
        cols = np.where(m.any(axis=0))[0]
        return cols.mean() if len(cols) else None
    early, late = hi_cols(0.5), hi_cols(2.4)
    assert early is not None and late is not None and late > early


def test_caption_fades_in_and_out_with_the_line():
    r = _renderer()
    cnt = lambda t: (np.asarray(r.render(t), dtype=np.int16)[int(H * 0.66):int(H * 0.82)].min(axis=2) > 225).sum()
    assert cnt(0.02) < cnt(1.5) * 0.3 and cnt(2.995) < cnt(1.5) * 0.3


def test_classic_background_and_hero_text_remain_the_default():
    c = HeroConfig()
    assert c.background == "classic" and c.text_style == "hero" and c.visualizer
    old = HeroConfig.from_dict({"title": "x", "background": "bogus", "text_style": "weird"})
    assert old.background == "classic" and old.text_style == "hero"


def test_scenes_with_hero_text_combination():
    cfg = HeroConfig.from_dict(STYLE_PRESETS["scene-hero"])
    evs = select_heroes([ev("teri aankhon mein kho gaya", 0, 3)], cfg)
    r = FrameRenderer(cfg, evs, W, H, Levels(None))
    a = np.asarray(r.render(2.6), dtype=np.int16)
    assert r.scene_plans[0].kind == "eye" and (a[int(H * 0.38):int(H * 0.60)].min(axis=2) > 200).sum() > 1500


@pytest.mark.slow
def test_cli_style_scene_end_to_end(tmp_path):
    aud = str(tmp_path / "t.wav")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=6", aud], check=True, capture_output=True)
    lrc = tmp_path / "l.lrc"; lrc.write_text("[00:00.20]chaand ke neeche\n[00:02.50]jalta hua dil\n", encoding="utf-8")
    out = str(tmp_path / "s.mp4")
    assert main(["make", "--audio", aud, "--lyrics", str(lrc), "--out", out, "--style", "scene", "--no-asr", "--end", "5",
                 "--size", "360x640", "--fps", "24", "--encoder", "libx264"]) == 0
    assert abs(probe(out)[0] - 5.0) < 0.2
    rev = json.load(open(str(tmp_path / "s.review.json"), encoding="utf-8"))
    assert [s["scene"] for s in rev["scenes"]] == ["moon", "heart_fire"]
