from ai_lyrics_hero.config import HeroConfig
from ai_lyrics_hero.hero import select_heroes, hero_color, normalize
from helpers import ev


def hero_text(e):
    return " ".join(w.text for w in e.words[e.hero[0]:e.hero[1] + 1]) if e.hero else None


def test_picks_content_word_not_stopword():
    e = select_heroes([ev("Maine na yeh jaana")], HeroConfig())[0]
    assert hero_text(e) == "jaana"


def test_short_hero_extends_to_two_word_phrase():
    e = select_heroes([ev("Are re are yeh kya hua")], HeroConfig())[0]
    assert hero_text(e) == "kya hua"
    e = select_heroes([ev("Are re are yeh kya hua")], HeroConfig(max_phrase_words=1))[0]
    assert hero_text(e) == "hua"


def test_single_word_event_is_hero():
    assert hero_text(select_heroes([ev("Forever")], HeroConfig())[0]) == "Forever"


def test_chant_becomes_one_phrase():
    e = select_heroes([ev("la la la la la")], HeroConfig())[0]
    assert e.hero == (0, 4)


def test_explicit_hero_wins_over_automatic():
    e = select_heroes([ev("burning down the house", explicit=(2, 3))], HeroConfig())[0]
    assert hero_text(e) == "the house"


def test_no_content_word_falls_back_to_no_hero():
    e = select_heroes([ev("are re are na to hai")], HeroConfig())[0]
    assert e.hero is None


def test_short_events_and_disabled_have_no_hero():
    assert select_heroes([ev("beautiful morning", end=0.3)], HeroConfig())[0].hero is None
    assert select_heroes([ev("beautiful morning")], HeroConfig(enabled=False))[0].hero is None


def test_repeated_word_not_hero_every_time_when_alternatives_exist():
    evs = select_heroes([ev("my heart heart tonight") for _ in range(1)] +
                        [ev("tonight my heart is free"), ev("tonight my heart is free")], HeroConfig(max_phrase_words=1))
    picks = [hero_text(e) for e in evs]
    assert len(set(picks)) > 1


def test_devanagari_and_romanized_hindi():
    e = select_heroes([ev("मैंने ना ये जाना")], HeroConfig())[0]
    assert hero_text(e).endswith("जाना")          # short word: widened into a 2-word phrase
    e = select_heroes([ev("koi na pehchaana")], HeroConfig())[0]
    assert hero_text(e) == "pehchaana"
    assert normalize("Pehchaana!,") == "pehchaana"


def test_unreliable_timing_still_selects_deterministically():
    a = select_heroes([ev("koi na pehchaana", reliable=False)], HeroConfig())[0].hero
    b = select_heroes([ev("koi na pehchaana", reliable=False)], HeroConfig())[0].hero
    assert a == b == (2, 2)


def test_color_is_deterministic_and_configurable():
    e = select_heroes([ev("koi na pehchaana")], HeroConfig())[0]
    assert hero_color(e, HeroConfig()) == hero_color(e, HeroConfig())
    assert hero_color(e, HeroConfig(highlight_color="gold")) == (255, 201, 77)
    assert hero_color(e, HeroConfig(highlight_color="#102030")) == (16, 32, 48)


def test_empty_input_does_not_crash():
    assert select_heroes([], HeroConfig()) == []
