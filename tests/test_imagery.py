import json
from ai_lyrics_hero.config import HeroConfig
from ai_lyrics_hero.imagery import Lexicon, plan_cues, load_sprite
from ai_lyrics_hero.image_anim import image_state, choose_preset, PRESETS, ORDER
from helpers import ev

L = Lexicon()


def ids(text, allow=False):
    return [m.entry_id for m in L.match(text.split(), allow)[0]]


def test_car_gets_car_never_something_else():
    assert ids("I see a car") == ["car"] and ids("meri gaadi") == ["car"]
    assert "bike" not in ids("I see a car")


def test_hindi_kar_does_not_become_car_and_bar_is_not_a_bar():
    assert ids("kar de mujhe maaf") == []
    assert ids("dil bar") == ["dilbar"]                    # phrase wins; 'bar' alone never imaged
    assert ids("a bar tonight") == []


def test_phrases_and_variants():
    assert ids("Blue eyes") == ["blue_eyes"] and ids("dilbar") == ["dilbar"]
    assert ids("teri aankhen") == ["eyes"] and ids("teri ankhein") == ["eyes"]
    assert ids("तेरी आँखें") == ["eyes"] and ids("दिल") == ["heart"]


def test_ambiguous_words_need_opt_in():
    ms, skipped = L.match("my ring".split())
    assert ms == [] and skipped == ["ring"]
    assert [m.entry_id for m in L.match("my ring".split(), True)[0]] == ["ring"]
    assert ids("sun mere humsafar") == []                  # Hindi 'sun' = listen: no sun picture


def test_unknown_words_get_no_image_no_fuzzy_guess():
    assert ids("carr bikee tonite") == []


def test_user_lexicon_extends_and_blocks(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"entries": [{"id": "bus", "emoji": "🚌", "forms": ["bus", "bas"]}], "block": ["heart"]}))
    L2 = Lexicon(str(p))
    assert [m.entry_id for m in L2.match(["bus"])[0]] == ["bus"]
    assert L2.match(["heart"])[0] == []


def test_plan_one_cue_per_line_with_correct_time_and_no_repeat():
    cfg = HeroConfig()
    evs = [ev("my car is red", 0, 3), ev("car car car", 3, 6), ev("under the moon", 6, 9), ev("nothing here", 9, 12)]
    cues, notes = plan_cues(evs, L, cfg)
    assert cues[0].entry_id == "car" and abs(cues[0].time - evs[0].words[1].start) < 1e-6
    assert 1 not in cues and any("recently" in n for n in notes)            # same picture not repeated
    assert cues[2].entry_id == "moon" and 3 not in cues


def test_same_emoji_from_different_entries_not_repeated():
    cfg = HeroConfig()
    evs = [ev("blue eyes", 0, 3), ev("teri aankhen", 3, 6)]
    cues, _ = plan_cues(evs, L, cfg)
    assert 0 in cues and 1 not in cues


def test_sprite_loads_and_user_png_overrides(tmp_path):
    from PIL import Image
    s = load_sprite("car", "🚗", 200, "")
    assert max(s.size) == 200 and s.mode == "RGBA" and s.getchannel("A").getextrema()[1] == 255
    Image.new("RGBA", (50, 20), (255, 0, 0, 255)).save(tmp_path / "car.png")
    o = load_sprite("car", "🚗", 100, str(tmp_path))
    assert o.size == (100, 40)


def test_animation_presets_start_hidden_or_offscreen_and_end_at_rest():
    for name in ORDER:
        s0 = image_state(0.0, name, 1080, 1920, 1, 1.0)
        assert s0.alpha < 0.5 or abs(s0.dx) > 100 or abs(s0.dy) > 100 or s0.scale < 0.7, name
        end = image_state(PRESETS[name][1] + 0.01, name, 1080, 1920, 1, 1.0)
        assert (abs(end.dx), abs(end.dy), end.scale, end.rot, end.alpha) == (0, 0, 1.0, 0, 1.0) or \
            (abs(end.dx) < 1e-6 and abs(end.dy) < 1e-6 and abs(end.scale - 1) < 1e-6 and abs(end.rot) < 1e-6 and end.alpha == 1.0), name
        assert image_state(-0.1, name, 1080, 1920).alpha == 0


def test_pop_overshoots_and_throw_comes_from_outside_frame():
    mid = [image_state(i * 0.01, "pop", 1080, 1920).scale for i in range(50)]
    assert max(mid) > 1.0 and abs(mid[-1] - 1.0) < 1e-6
    assert abs(image_state(0.0, "throw", 1080, 1920, 1).dx) > 1080 * 0.7


def test_preset_choice_deterministic_and_no_immediate_repeat():
    assert choose_preset("car", None) == choose_preset("car", None)
    for k in ["car", "moon", "heart", "eyes"]:
        p = choose_preset(k, None)
        assert choose_preset(k, p) != p
    assert choose_preset("car", None, intensity=0) == "pop"
