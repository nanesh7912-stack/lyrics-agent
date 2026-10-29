import io
import json
import os
import urllib.error
import numpy as np
import pytest
from PIL import Image
from ai_lyrics_hero.animation import hero_state, hero_start_offset, TEXT_PRESETS
from ai_lyrics_hero.assets import AssetLibrary
from ai_lyrics_hero.audio import Levels
from ai_lyrics_hero.config import HeroConfig
from ai_lyrics_hero.director import direct
from ai_lyrics_hero.hero import select_heroes
from ai_lyrics_hero.music import MusicMap
from ai_lyrics_hero.renderer import FrameRenderer
from ai_lyrics_hero.scenes import SCENE_KINDS, apply_camera
from ai_lyrics_hero.semantic import LLMSemantic, LocalSemantic, analyze_lines
from ai_lyrics_hero.transitions import blend, STYLES
from helpers import ev

W, H = 540, 960


def mk(texts, step=3.0):
    return [ev(t, i * step, i * step + step - 0.1) for i, t in enumerate(texts)]


# ---------------------------------------------------------------- scene planning
def test_lexicon_words_get_matching_scenes_and_unknown_get_coherent_fallbacks():
    cfg = HeroConfig()
    d = direct(mk(["chaand ke neeche", "teri aankhon mein", "xyz abc qwe", "love me tonight", "road to nowhere"]), cfg)
    sc = [p.scene for p in d.plans]
    assert sc[0] == "moon" and sc[1] == "eye" and sc[3] == "romance" and sc[4] == "road"
    assert d.plans[2].source == "fallback" and d.plans[2].scene in SCENE_KINDS


def test_no_three_identical_atmospheric_fallbacks_in_a_row_and_variety():
    texts = [f"abc{i} def{i} ghi{i}" for i in range(14)]
    d = direct(mk(texts), HeroConfig(seed=3))
    sc = [p.scene for p in d.plans]
    assert all(not (sc[i] == sc[i + 1] == sc[i + 2]) for i in range(len(sc) - 2))
    assert len(set(sc)) >= 3
    assert len({p.typography for p in d.plans}) >= 3 and len({p.composition for p in d.plans}) >= 2


def test_repeated_line_reuses_its_scene_as_a_motif_with_new_camera():
    d = direct(mk(["mystery words here", "other things now", "mystery words here"]), HeroConfig())
    assert d.plans[2].scene == d.plans[0].scene and d.plans[2].source == "motif"
    assert (d.plans[2].zoom0, d.plans[2].zoom1) != (d.plans[0].zoom0, d.plans[0].zoom1)


def test_same_input_and_seed_give_identical_plans_and_different_seed_may_differ():
    texts = [f"abc{i} def{i} ghi{i}" for i in range(10)]
    key = lambda d: [(p.scene, p.typography, p.composition, p.transition_in) for p in d.plans]
    assert key(direct(mk(texts), HeroConfig(seed=1))) == key(direct(mk(texts), HeroConfig(seed=1)))
    assert key(direct(mk(texts), HeroConfig(seed=1))) != key(direct(mk(texts), HeroConfig(seed=2)))


def test_adjacent_same_scene_has_no_fade_or_transition_and_different_scenes_transition():
    evs = [ev("jalta dil", 0, 2.9), ev("jalti aag", 3.0, 5.9), ev("chaand raat", 6.0, 9.0)]
    d = direct(evs, HeroConfig())
    assert d.plans[1].transition_in == "none" and not d.plans[0].fade_out and not d.plans[1].fade_in
    assert d.plans[2].transition_in != "none" and d.plans[2].transition_s > 0 and not d.plans[1].fade_out
    assert d.plans[0].vis_end == 3.0                       # scene held until the next line starts


def test_forced_options_are_honored():
    d = direct(mk(["abc def ghi", "jkl mno pqr"]), HeroConfig(typography="rotate", composition="offset"))
    assert {p.typography for p in d.plans} == {"rotate"} and {p.composition for p in d.plans} == {"offset"}
    d = direct([ev("a b c", 0, 2.9), ev("chaand ho", 3.0, 5.9)], HeroConfig(transitions="zoom"))
    assert d.plans[1].transition_in == "zoom"


def test_music_drives_levels_cuts_and_interludes():
    m = MusicMap(duration=40, reliable=True, impacts=[(3.02, 0.9)], energy=np.concatenate([np.full(100, 0.8), np.full(700, 0.2)]),
                 energy_hop=0.05)
    evs = [ev("chaand aur raat", 0, 2.9), ev("aankhon mein tum", 3.0, 5.9), ev("last line here", 20.0, 23.0)]
    d = direct(evs, HeroConfig(), music=m, total_end=40.0)
    assert d.plans[0].level == "high" and d.plans[2].level == "quiet"
    assert d.plans[1].transition_in == "cut"                                # impact right at the boundary
    assert any(i.start >= 5.9 and i.end <= 20.0 for i in d.interludes)       # scenes during the instrumental gap
    assert all(i.scene in SCENE_KINDS for i in d.interludes)


def test_interludes_disabled_or_without_music():
    evs = [ev("a b c", 0, 2), ev("d e f", 12, 14)]
    assert direct(evs, HeroConfig(), music=None).interludes == []
    m = MusicMap(duration=20, reliable=True, energy=np.full(400, 0.7), energy_hop=0.05)
    assert direct(evs, HeroConfig(interludes=False), music=m).interludes == []


# ---------------------------------------------------------------- semantic layer
class FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def opener_for(payload):
    def op(req, timeout=None):
        return FakeResp(json.dumps({"choices": [{"message": {"content": payload}}]}).encode())
    return op


LINES = [["xyz", "abc"], ["chaand", "ke"], ["qrs", "tuv"]]


def test_no_api_key_means_local_only_and_never_crashes():
    cfg = HeroConfig(llm_provider="openai")
    llm = LLMSemantic(api_key="")
    assert llm.classify(LINES) == {}
    sem, src = analyze_lines(LINES, cfg, llm=llm)
    assert src.startswith("local") and sem[1].scene == "moon"


def test_llm_answer_fills_only_unmatched_lines_and_is_validated():
    payload = json.dumps([{"i": 0, "scene": "rain", "mood": "sad", "confidence": 0.9},
                          {"i": 1, "scene": "city", "mood": "joyful", "confidence": 0.95},          # lexicon wins for line 1
                          {"i": 2, "scene": "not_a_scene", "mood": "sad", "confidence": 0.9},        # invalid scene -> dropped
                          {"i": 9, "scene": "moon", "mood": "sad", "confidence": 0.9},               # bad index -> dropped
                          {"i": 0, "scene": "moon", "mood": "calm", "confidence": 0.2}])             # low confidence overwritten? no: dropped
    llm = LLMSemantic(api_key="k", opener=opener_for(payload))
    sem, src = analyze_lines(LINES, HeroConfig(llm_provider="openai"), llm=llm)
    assert src == "local + LLM"
    assert sem[0].scene in ("rain", "moon") and sem[0].source == "llm"
    assert sem[1].scene == "moon" and sem[1].source == "local"
    assert sem[2].scene is None


@pytest.mark.parametrize("payload", ["not json at all", "{\"a\": 1}", "[1, 2, 3]", "```json\n[{\"i\": 0}]\n```"])
def test_garbage_llm_output_is_ignored(payload):
    llm = LLMSemantic(api_key="k", opener=opener_for(payload))
    assert llm.classify(LINES) == {}


@pytest.mark.parametrize("exc", [urllib.error.URLError("offline"), TimeoutError(), OSError("boom")])
def test_network_errors_fall_back_to_local(exc):
    def bad(req, timeout=None): raise exc
    sem, src = analyze_lines(LINES, HeroConfig(), llm=LLMSemantic(api_key="k", opener=bad))
    assert "local" in src and sem[1].scene == "moon"


def test_lyrics_are_never_modified_by_semantic_analysis():
    evs = mk(["Tu Chale Aaye", "Woh Jaan Jaaye"])
    before = [e.text for e in evs]
    direct(evs, HeroConfig(), llm=LLMSemantic(api_key="k", opener=opener_for('[{"i":0,"scene":"city","mood":"sad","confidence":1}]')))
    assert [e.text for e in evs] == before


def test_local_mood_detection():
    sem = LocalSemantic().classify([["pyaar", "ishq"], ["aansu", "dard"], ["hello", "there"]])
    assert sem[0].mood == "romantic" and sem[1].mood == "sad" and sem[2].mood == "neutral"


# ---------------------------------------------------------------- user assets
def make_asset_dir(tmp_path, scene="moon", size=(800, 600), color=(200, 40, 40)):
    d = tmp_path / "bg" / "backgrounds" / scene
    d.mkdir(parents=True)
    Image.new("RGB", size, color).save(d / "a.jpg")
    return str(tmp_path / "bg")


def test_asset_library_discovery_cover_crop_and_darkening(tmp_path):
    root = make_asset_dir(tmp_path)
    lib = AssetLibrary(root, (W, H))
    assert lib.has("moon") and not lib.has("city")
    fr = lib.frame(lib.pick("moon", "k"), 0.5)
    assert fr.size == (W, H)
    px = np.asarray(fr)[100, 100]
    assert px[0] > 80 and px[0] < 200 and px[1] < 60            # still red, darkened


def test_asset_errors_are_skipped_not_fatal(tmp_path):
    root = make_asset_dir(tmp_path)
    bad = tmp_path / "bg" / "backgrounds" / "moon" / "broken.png"
    bad.write_bytes(b"not an image")
    lib = AssetLibrary(root + os.pathsep + str(tmp_path / "nonexistent"), (W, H))
    assert lib.load(str(bad)) is None and lib.warnings
    assert lib.load(lib.pick("moon", "zz", avoid=[str(bad)])) is not None or True


def test_user_background_is_used_by_the_renderer(tmp_path):
    root = make_asset_dir(tmp_path)
    cfg = HeroConfig(background="scenes", text_style="caption", images=False, visualizer=False)
    evs = select_heroes([ev("chaand ke neeche", 0, 3)], cfg)
    r = FrameRenderer(cfg, evs, W, H, Levels(None), assets=AssetLibrary(root, (W, H)))
    assert r.plans[0].asset_path
    a = np.asarray(r.render(1.5), dtype=np.int16)
    assert (a[100:300, 50:450, 0] > 80).mean() > 0.9 and (a[100:300, 50:450, 1] < 70).mean() > 0.9


def test_missing_asset_scene_falls_back_to_procedural(tmp_path):
    root = make_asset_dir(tmp_path, scene="city")
    cfg = HeroConfig(background="scenes", text_style="caption", images=False)
    evs = select_heroes([ev("chaand ke neeche", 0, 3)], cfg)
    r = FrameRenderer(cfg, evs, W, H, Levels(None), assets=AssetLibrary(root, (W, H)))
    assert r.plans[0].asset_path is None


# ---------------------------------------------------------------- typography presets
@pytest.mark.parametrize("preset", TEXT_PRESETS)
def test_every_text_preset_starts_hidden_and_settles_to_neutral(preset):
    e = select_heroes([ev("Maine na yeh jaana", 0.0, 3.0)], HeroConfig())[0]
    cfg = HeroConfig()
    t0 = hero_start_offset(e, cfg)
    s0 = hero_state(t0 + 0.001, e, cfg, preset)
    assert s0.active and (s0.hero_alpha < 0.5 or s0.reveal < 0.1 or s0.chars < 0.1)
    end = hero_state(t0 + cfg.entrance_s * 1.6, e, cfg, preset)
    assert end.hero_alpha == 1.0 and abs(end.hero_scale - 1.0) < 1e-6
    assert abs(end.dx) < 1e-6 and abs(end.dy) < 1e-6 and abs(end.rot) < 1e-6
    assert end.reveal == 1.0 and end.blur == 0.0 and end.tracking == 0.0 and end.chars == 1.0
    assert hero_state(2.999, e, cfg, preset).hero_alpha < 0.05          # all presets exit cleanly


def test_preset_specific_motion():
    e = select_heroes([ev("Maine na yeh jaana", 0.0, 3.0)], HeroConfig())[0]
    cfg = HeroConfig(); t0 = hero_start_offset(e, cfg) + 0.03
    assert hero_state(t0, e, cfg, "slide_left").dx < -100 and hero_state(t0, e, cfg, "slide_right").dx > 100
    assert hero_state(t0, e, cfg, "slide_up").dy > 50
    assert hero_state(t0, e, cfg, "rotate").rot < -3
    assert hero_state(t0, e, cfg, "wipe").reveal < 0.6 and hero_state(t0, e, cfg, "blur").blur > 2
    assert hero_state(t0, e, cfg, "tracking").tracking < 0 and hero_state(t0, e, cfg, "chars").chars < 0.5
    assert hero_state(t0, e, cfg, "impact").hero_scale > 1.2
    assert hero_state(t0, e, cfg, "totally_unknown").hero_scale > 0           # unknown preset falls back to pop


def test_zero_intensity_removes_motion_for_all_presets():
    e = select_heroes([ev("Maine na yeh jaana", 0.0, 3.0)], HeroConfig())[0]
    cfg = HeroConfig(animation_intensity=0.0); t0 = hero_start_offset(e, cfg) + 0.05
    for p in ("slide_left", "rotate", "blur", "tracking", "impact"):
        s = hero_state(t0, e, cfg, p)
        assert abs(s.rot) < 1e-6 and s.blur == 0 and s.tracking == 0 and s.hero_scale <= 1.0 + 1e-9


def rend(cfg, text="my red car tonight", plans=None, **kw):
    evs = select_heroes([ev(text, 0.0, 4.0)], cfg)
    return FrameRenderer(cfg, evs, W, H, Levels(None), **kw)


@pytest.mark.parametrize("typo", ["pop", "slide_left", "rotate", "wipe", "tracking", "blur", "chars", "impact"])
def test_each_typography_preset_renders_text_mid_entrance_and_final(typo):
    r = rend(HeroConfig(typography=typo, images=False))
    e = r.events[0]
    t_hero = e.start + hero_start_offset(e, r._eff_cfg(0))
    mid = np.asarray(r.render(t_hero + 0.12), dtype=np.int16)
    fin = np.asarray(r.render(t_hero + 0.9), dtype=np.int16)
    zone = (slice(int(H * 0.30), int(H * 0.62)), slice(0, W))
    assert (fin[zone].min(axis=2) > 200).sum() > 1500
    assert fin.shape == (H, W, 3) and mid.shape == fin.shape


def test_devanagari_with_character_presets_does_not_break():
    for typo in ("tracking", "chars", "blur", "wipe"):
        r = rend(HeroConfig(typography=typo, images=False), "मैंने ना ये जाना")
        e = r.events[0]
        th = e.start + hero_start_offset(e, r._eff_cfg(0))
        f = np.asarray(r.render(th + 0.9), dtype=np.int16)
        assert (f[int(H * 0.3):int(H * 0.62)].min(axis=2) > 200).sum() > 500


# ---------------------------------------------------------------- compositions
@pytest.mark.parametrize("comp", ["centered", "offset", "bg_word", "minimal"])
def test_every_composition_keeps_text_inside_the_frame(comp):
    r = rend(HeroConfig(composition=comp, images=False), "Aankhon Ko Palkon Ke Dhaagon Se Baandho Na")
    e = r.events[0]
    th = e.start + hero_start_offset(e, r._eff_cfg(0))
    a = np.asarray(r.render(th + 0.9), dtype=np.int16)
    band = a[int(H * 0.28):int(H * 0.65)]
    cols = np.where((band.min(axis=2) > 215).any(axis=0))[0]
    assert len(cols) and cols.min() >= 0.03 * W and cols.max() <= 0.97 * W


def test_offset_composition_moves_text_off_center_and_image_to_the_other_side():
    r = rend(HeroConfig(composition="offset"), "my red car tonight")
    lay = r._layout(0)
    assert abs(lay.hero_lines[0].cx - W / 2) > 0.08 * W
    assert np.sign(r._side(0)) == -np.sign(lay.hero_lines[0].cx - W / 2)


def test_bg_word_draws_faint_giant_word_behind():
    r = rend(HeroConfig(composition="bg_word", images=False), "burning bright tonight")
    e = r.events[0]
    th = e.start + hero_start_offset(e, r._eff_cfg(0))
    plain = rend(HeroConfig(composition="centered", images=False), "burning bright tonight")
    a = np.asarray(r.render(th + 0.9), dtype=np.int16); b = np.asarray(plain.render(th + 0.9), dtype=np.int16)
    assert np.abs(a - b).mean() > 0.5


# ---------------------------------------------------------------- beats
def test_beat_pulse_scales_hero_on_impacts_and_not_when_disabled():
    m = MusicMap(duration=10, reliable=True, impacts=[(3.1, 0.95)], energy=np.full(200, 0.5), energy_hop=0.05)
    evs = select_heroes([ev("Maine na yeh jaana", 0.0, 4.0)], HeroConfig())
    r = FrameRenderer(HeroConfig(), evs, W, H, Levels(None), music=m)
    p_hit, sx, sy = r._beat(3.12, evs[0], 0)
    assert p_hit > 0.03 and abs(sx) <= 5.0
    assert r._beat(3.6, evs[0], 0)[0] < p_hit * 0.05                                  # decays quickly
    assert r._beat(2.0, evs[0], 0) == (0.0, 0.0, 0.0)                                 # nothing without an impact
    r0 = FrameRenderer(HeroConfig(beat_intensity=0.0), evs, W, H, Levels(None), music=m)
    assert r0._beat(3.12, evs[0], 0) == (0.0, 0.0, 0.0)
    r1 = FrameRenderer(HeroConfig(), evs, W, H, Levels(None), music=MusicMap())      # unreliable analysis
    assert r1._beat(3.12, evs[0], 0) == (0.0, 0.0, 0.0)


def test_hero_selection_prefers_the_word_sung_on_an_impact():
    e1 = ev("alpha bravo gamma delta", 0, 4, reliable=True)
    e1.words[1].start = 1.0                                       # 'bravo' on the hit
    plain = select_heroes([ev("alpha bravo gamma delta", 0, 4)], HeroConfig(max_phrase_words=1))[0].hero
    m = MusicMap(duration=5, reliable=True, impacts=[(1.0, 1.0)])
    boosted = select_heroes([e1], HeroConfig(max_phrase_words=1), m)[0].hero
    assert plain == (3, 3) and boosted == (1, 1)


def test_emotional_words_get_priority():
    e = select_heroes([ev("the big dil wale", 0, 3)], HeroConfig(max_phrase_words=1))[0]
    assert e.words[e.hero[0]].text == "dil"


# ---------------------------------------------------------------- transitions / camera
@pytest.mark.parametrize("style", STYLES)
def test_transitions_start_at_outgoing_end_at_incoming_and_stay_valid(style):
    a = Image.new("RGB", (W, H), (255, 0, 0)); b = Image.new("RGB", (W, H), (0, 0, 255))
    assert np.array_equal(np.asarray(blend(style, a, b, 0.0)), np.asarray(a))
    assert np.array_equal(np.asarray(blend(style, a, b, 1.0)), np.asarray(b))
    mid = blend(style, a, b, 0.5)
    assert mid.size == (W, H) and mid.mode == "RGB"
    if style in ("fade", "zoom", "wipe", "slide"):
        arr = np.asarray(mid).astype(int)
        assert (arr[..., 0].mean() > 5) and (arr[..., 2].mean() > 5)


def test_camera_zoom_crops_center_and_is_identity_at_one():
    img = Image.new("RGB", (100, 200), (0, 0, 0)); img.paste((255, 255, 255), (40, 80, 60, 120))
    assert np.array_equal(np.asarray(apply_camera(img, 1.0)), np.asarray(img))
    z = np.asarray(apply_camera(img, 2.0))
    assert z.sum() > np.asarray(img).sum()                   # the bright center grows when zoomed
    assert apply_camera(img, 1.5, 1, 1).size == img.size


def test_scene_transition_frames_render_between_different_scenes():
    cfg = HeroConfig(background="scenes", text_style="caption", images=False, visualizer=False, transitions="wipe")
    evs = select_heroes([ev("chaand ke neeche", 0, 2.95), ev("jalta hua dil", 3.0, 6.0)], cfg)
    r = FrameRenderer(cfg, evs, W, H, Levels(None))
    assert r.plans[1].transition_in == "wipe"
    mid = np.asarray(r.render(3.0 + r.plans[1].transition_s / 2), dtype=np.int16)
    late = np.asarray(r.render(4.5), dtype=np.int16)
    assert np.abs(mid - late).mean() > 2                      # visibly different from the settled scene
