"""Configuration with backward-compatible defaults.

Older config dicts that lack the new keys (or contain unknown keys) still load.
"""
from dataclasses import dataclass, fields, asdict

PALETTE = {
    "white": (255, 255, 255),
    "pink": (255, 90, 150),
    "cyan": (95, 240, 255),
    "gold": (255, 201, 77),
}
OUTRO_AUDIO_MODES = ("continue", "outro", "silent", "mix")
COMPOSITIONS = ("auto", "centered", "offset", "bg_word", "minimal")
TYPO_PRESETS = ("auto", "pop", "slide", "rotate", "wipe", "tracking", "blur", "chars", "impact")
TRANSITIONS = ("auto", "fade", "zoom", "slide", "wipe", "cut")
AUTO_CYCLE = ("pink", "cyan", "gold", "white")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class HeroConfig:
    enabled: bool = True
    support_size: int = 60          # px at 1080 px width
    hero_size: int = 210            # maximum px at 1080 px width
    highlight_color: str = "auto"   # auto | white | pink | cyan | gold | #rrggbb
    glow_intensity: float = 0.8     # 0..1.5
    entrance_s: float = 0.35
    exit_s: float = 0.18
    animation_intensity: float = 1.0  # 0 = no overshoot/motion, 1 = default, 2 = strong
    max_phrase_words: int = 2
    uppercase: bool = False         # False = show the lyric exactly as written (case preserved)
    min_event_s: float = 0.6        # events shorter than this never get a hero word
    title: str = ""
    images: bool = True              # meaning-matched illustrations (exact dictionary matches only)
    image_size: int = 300            # px at 1080 px width
    allow_ambiguous: bool = False    # show images for words with several meanings (e.g. Hindi 'sun')
    image_gap_events: int = 2        # don't repeat the same picture within this many lyric lines
    lexicon_path: str = ""           # optional JSON with extra/override dictionary entries
    icons_dir: str = ""              # optional folder of transparent PNGs named <entry id>.png
    background: str = "classic"      # classic (purple geometric) | scenes (meaning-driven animated night scenes)
    text_style: str = "hero"         # hero (big glowing word) | caption (small bold lines, current word highlighted)
    visualizer: bool = True          # audio bars at the bottom
    caption_size: int = 66           # px at 1080 px width (caption style)
    caption_highlight: str = "#9aa8ff"
    scene_lexicon_path: str = ""     # optional JSON extending scenes.json
    target_duration: float = 0.0     # final video length incl. outro in seconds (0 = derive from audio / --end)
    outro_duration: float = 0.0      # seconds of the outro to use (0 = the outro file's full length)
    outro_audio: str = "continue"    # continue | outro | silent | mix
    crossfade_s: float = 1.0         # used by outro_audio=mix
    seed: int = 0                    # makes every random-looking choice reproducible
    verify_output: bool = True       # ffprobe/decode check after rendering
    beat_intensity: float = 1.0      # 0 disables beat-driven pulses/shake, 2 = strong
    asset_dirs: str = ""             # folders with your own images (see assets/user/README.txt), os.pathsep separated
    llm_provider: str = "none"       # none | openai (OpenAI-compatible endpoint via AI_LYRICS_LLM_* env vars)
    composition: str = "auto"        # auto | centered | offset | bg_word | minimal
    typography: str = "auto"         # auto | pop | slide | rotate | wipe | tracking | blur | chars | impact
    transitions: str = "auto"        # auto | fade | zoom | slide | wipe | cut
    interludes: bool = True          # show scenes (never invented lyrics) during instrumental breaks

    @classmethod
    def from_dict(cls, data=None):
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        defaults = cls()
        kwargs = {}
        for k, v in data.items():
            if k not in known:
                continue  # ignore unknown / future keys
            typ = type(getattr(defaults, k))
            try:
                if typ is bool:
                    if isinstance(v, bool):
                        kwargs[k] = v
                    elif isinstance(v, str):
                        value = v.strip().lower()
                        if value in ("true", "1", "yes", "on"):
                            kwargs[k] = True
                        elif value in ("false", "0", "no", "off"):
                            kwargs[k] = False
                        else:
                            continue
                    elif isinstance(v, (int, float)) and v in (0, 1):
                        kwargs[k] = bool(v)
                    else:
                        continue
                else:
                    kwargs[k] = typ(v)
            except (TypeError, ValueError):
                continue  # keep default for unusable values
        cfg = cls(**kwargs)
        cfg.support_size = int(_clamp(cfg.support_size, 24, 140))
        cfg.hero_size = int(_clamp(cfg.hero_size, 60, 420))
        cfg.glow_intensity = _clamp(cfg.glow_intensity, 0.0, 1.5)
        cfg.entrance_s = _clamp(cfg.entrance_s, 0.05, 1.5)
        cfg.exit_s = _clamp(cfg.exit_s, 0.0, 1.0)
        cfg.animation_intensity = _clamp(cfg.animation_intensity, 0.0, 2.0)
        cfg.max_phrase_words = int(_clamp(cfg.max_phrase_words, 1, 4))
        cfg.image_size = int(_clamp(cfg.image_size, 80, 600))
        cfg.caption_size = int(_clamp(cfg.caption_size, 24, 140))
        cfg.target_duration = max(0.0, cfg.target_duration)
        cfg.outro_duration = max(0.0, cfg.outro_duration)
        cfg.crossfade_s = _clamp(cfg.crossfade_s, 0.0, 10.0)
        if cfg.outro_audio not in OUTRO_AUDIO_MODES:
            cfg.outro_audio = "continue"
        cfg.beat_intensity = _clamp(cfg.beat_intensity, 0.0, 2.0)
        if cfg.llm_provider not in ("none", "openai"):
            cfg.llm_provider = "none"
        for k, ok in (("composition", COMPOSITIONS), ("typography", TYPO_PRESETS), ("transitions", TRANSITIONS)):
            if getattr(cfg, k) not in ok:
                setattr(cfg, k, "auto")
        if cfg.background not in ("classic", "scenes"):
            cfg.background = "classic"
        if cfg.text_style not in ("hero", "caption"):
            cfg.text_style = "hero"
        cfg.image_gap_events = int(_clamp(cfg.image_gap_events, 0, 20))
        return cfg

    def to_dict(self):
        return asdict(self)


def parse_color(spec, fallback="white"):
    spec = (spec or "").strip().lower()
    if spec in PALETTE:
        return PALETTE[spec]
    if len(spec) == 7 and spec.startswith("#"):
        try:
            return tuple(int(spec[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            pass
    return PALETTE[fallback]


STYLE_PRESETS = {
    "classic": {},
    "hero": {"background": "classic", "text_style": "hero"},
    "scene": {"background": "scenes", "text_style": "caption", "images": False, "visualizer": False},
    "scene-hero": {"background": "scenes", "text_style": "hero", "images": False, "visualizer": False},
}
