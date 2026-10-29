from ai_lyrics_hero.config import HeroConfig, parse_color


def test_defaults_work_without_any_setup():
    c = HeroConfig.from_dict({})
    assert c.enabled and c.hero_size > c.support_size


def test_old_preset_without_new_keys_still_loads():
    old = {"title": "My Song", "theme": "purple", "fps": 30}   # unknown keys ignored
    c = HeroConfig.from_dict(old)
    assert c.title == "My Song" and c.enabled and c.glow_intensity == HeroConfig().glow_intensity


def test_bad_values_fall_back_or_clamp():
    c = HeroConfig.from_dict({"hero_size": "abc", "glow_intensity": 99, "entrance_s": -3})
    assert c.hero_size == HeroConfig().hero_size
    assert c.glow_intensity == 1.5 and c.entrance_s == 0.05


def test_roundtrip_and_colors():
    c = HeroConfig.from_dict({"highlight_color": "cyan"})
    assert HeroConfig.from_dict(c.to_dict()) == c
    assert parse_color("#ff0000") == (255, 0, 0)
    assert parse_color("nonsense") == (255, 255, 255)


def test_shipped_example_configs_only_use_known_keys_and_load():
    import json, os
    from dataclasses import fields
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    known = {f.name for f in fields(HeroConfig)}
    for name in ("hero_config.example.json", "preset_scene.json"):
        data = json.load(open(os.path.join(root, name), encoding="utf-8"))
        assert set(data) <= known, (name, set(data) - known)
        HeroConfig.from_dict(data)


def test_new_options_validate_and_old_configs_keep_working():
    c = HeroConfig.from_dict({"outro_audio": "bogus", "typography": "nope", "composition": "x", "transitions": "y",
                              "llm_provider": "other", "beat_intensity": 9, "target_duration": -4})
    assert (c.outro_audio, c.typography, c.composition, c.transitions, c.llm_provider) == ("continue", "auto", "auto", "auto", "none")
    assert c.beat_intensity == 2.0 and c.target_duration == 0.0
    old = HeroConfig.from_dict({"hero_size": 200, "glow_intensity": 0.5, "title": "x"})
    assert old.background == "classic" and old.text_style == "hero" and old.outro_audio == "continue"
