"""Visual director: turns lyric events + music + dictionaries into a deterministic per-line visual plan.

Inputs never modify the lyrics. Same events + config + seed + music => identical plans."""
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .animation import hero_start_offset
from .assets import AssetLibrary
from .hero import normalize
from .scenes import SceneLexicon
from .semantic import analyze_lines

ADJACENT_GAP = 0.15
MOOD_SCENE = {"romantic": "romance", "sad": "night", "energetic": "streaks", "intense": "streaks",
              "calm": "night", "dreamy": "dream", "joyful": "nature"}
FALLBACK_POOL = {"quiet": ["night", "dream", "stars", "galaxy", "sky"],
                 "mid": ["sky", "stars", "dream", "galaxy", "night"],
                 "high": ["streaks", "galaxy", "stars", "dream"]}
MOOD_ACCENT = {"romantic": "pink", "sad": "cyan", "energetic": "gold", "intense": "gold", "joyful": "gold",
               "dreamy": "cyan", "calm": "white"}
TYPO_AUTO = {"quiet": ["blur", "tracking", "pop", "rotate"],
             "mid": ["pop", "wipe", "chars", "slide_left", "tracking", "slide_up", "rotate"],
             "high": ["slide_right", "rotate", "pop", "slide_left", "slide_down", "wipe"]}
TRANSITION_AUTO = {"quiet": ["fade"], "mid": ["fade", "zoom", "wipe"], "high": ["slide", "zoom", "wipe"]}
INTENSITY = {"quiet": 0.65, "mid": 1.0, "high": 1.25}
ZOOM = {"quiet": 0.03, "mid": 0.05, "high": 0.08}


@dataclass
class VisualPlan:
    index: int
    start: float
    end: float
    vis_end: float
    scene: str
    word: str = ""
    source: str = "fallback"          # lexicon | llm | motif | mood | fallback
    mood: str = "neutral"
    accent: Optional[str] = None
    level: str = "mid"
    intensity: float = 1.0
    zoom0: float = 1.0
    zoom1: float = 1.05
    pan: tuple = (1.0, 0.0)
    typography: str = "pop"
    composition: str = "centered"
    transition_in: str = "none"
    transition_s: float = 0.0
    fade_in: bool = True
    fade_out: bool = True
    asset_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def kind(self):                     # lets the plan be passed wherever a ScenePlan is expected
        return self.scene


@dataclass
class Interlude:
    start: float
    end: float
    scene: str
    level: str

    @property
    def kind(self):
        return self.scene


@dataclass
class Direction:
    plans: List[VisualPlan]
    interludes: List[Interlude]
    notes: List[str]
    semantic_source: str = "local"


def _pick(rnd, options, avoid):
    pool = [o for o in options if o not in avoid] or list(options)
    return pool[rnd.randrange(len(pool))]


def direct(events, cfg, music=None, lexicon=None, assets: Optional[AssetLibrary] = None, llm=None, total_end=None):
    notes = []
    lex = lexicon or SceneLexicon(cfg.scene_lexicon_path or None)
    lines = [[w.text for w in ev.words] for ev in events]
    sem, sem_src = analyze_lines(lines, cfg, lex, llm)
    plans: List[VisualPlan] = []
    seen_scene: Dict[str, str] = {}
    recent_scenes, recent_typo, recent_comp, recent_assets = [], [], [], []
    for i, ev in enumerate(events):
        rnd = random.Random(f"{cfg.seed}:{i}")
        mid_t = (ev.start + ev.end) / 2
        level = music.level_at(mid_t) if music is not None else "mid"
        key = " ".join(normalize(w.text) for w in ev.words)
        s = sem[i]
        p = VisualPlan(i, ev.start, ev.end, ev.end, "sky", level=level, mood=s.mood)
        # --- scene
        if s.scene and s.scene != "sky":
            p.scene, p.source = s.scene, ("llm" if s.source == "llm" else "lexicon")
            p.word = ""
        elif key in seen_scene:
            p.scene, p.source = seen_scene[key], "motif"                  # repeated line: same motif, new camera
            p.notes.append("repeated line reuses its earlier scene")
        else:
            if s.mood in MOOD_SCENE and s.confidence >= 0.5 and s.mood not in ("energetic", "intense"):
                p.scene, p.source = MOOD_SCENE[s.mood], "mood"
            else:
                pool = FALLBACK_POOL[level]
                p.scene = _pick(rnd, pool, recent_scenes[-2:])
                p.source = "fallback"
        if p.source in ("fallback", "mood") and len(recent_scenes) >= 2 and recent_scenes[-1] == recent_scenes[-2] == p.scene:
            p.scene = _pick(rnd, FALLBACK_POOL[level], recent_scenes[-2:])      # never 3 identical atmospheric scenes
        seen_scene.setdefault(key, p.scene)
        recent_scenes.append(p.scene)
        # --- camera
        z = ZOOM[level]
        inward = (i % 2 == 0) != (p.source == "motif")           # a repeated motif gets the opposite camera move
        p.zoom0, p.zoom1 = (1.0, 1.0 + z) if inward else (1.0 + z, 1.0)
        p.pan = ((-1) ** i, 0.5 * (-1) ** (i // 2))
        p.intensity = INTENSITY[level]
        # --- typography
        hero_len = 0
        if ev.hero:
            hero_len = sum(len(w.text) for w in ev.words[ev.hero[0]:ev.hero[1] + 1])
        if cfg.typography != "auto":
            t = cfg.typography
            if t == "slide":
                t = ["slide_left", "slide_right", "slide_up", "slide_down"][i % 4]
        else:
            impact = None
            if music is not None and ev.hero:
                hs = ev.start + hero_start_offset(ev, cfg)
                impact = music.nearest_impact(hs, 0.12, 0.12)
            if impact and impact[0] >= 0.0 and level != "quiet":
                t = "impact"
            else:
                opts = [o for o in TYPO_AUTO[level] if not (o == "chars" and not (3 <= hero_len <= 12))]
                t = _pick(rnd, opts, recent_typo[-2:])
        p.typography = t
        recent_typo.append(t)
        # --- composition
        if cfg.composition != "auto":
            p.composition = cfg.composition
        elif i == 0:
            p.composition = "centered"
        elif key in seen_scene and p.source == "motif" and hero_len >= 3 and level != "quiet":
            p.composition = "bg_word"
        elif level == "quiet":
            p.composition = "minimal"
        else:
            p.composition = _pick(rnd, ["centered", "offset", "bg_word" if (level == "high" and hero_len >= 3) else "centered"],
                                  recent_comp[-1:])
        recent_comp.append(p.composition)
        # --- palette
        p.accent = MOOD_ACCENT.get(s.mood)
        ev.accent = p.accent
        # --- user assets
        if assets is not None and assets.has(p.scene):
            path = assets.pick(p.scene, f"{cfg.seed}:{i}:{p.scene}", recent_assets[-3:])
            if path and assets.load(path) is not None:
                p.asset_path = path
                recent_assets.append(path)
        plans.append(p)
    # --- continuity and transitions
    for i, p in enumerate(plans):
        nxt = plans[i + 1] if i + 1 < len(plans) else None
        if nxt and nxt.start - p.end < ADJACENT_GAP:
            p.vis_end = nxt.start
        prev = plans[i - 1] if i > 0 else None
        adjacent = prev is not None and p.start - prev.end < ADJACENT_GAP
        same = adjacent and prev.scene == p.scene and prev.asset_path == p.asset_path
        if same:
            p.fade_in = False
            plans[i - 1].fade_out = False
        if adjacent and not same:
            rnd = random.Random(f"{cfg.seed}:t{i}")
            style = cfg.transitions
            near = music.nearest_impact(p.start, 0.15, 0.15) if music is not None else None
            if style == "auto":
                style = "cut" if (near and p.level != "quiet") else _pick(rnd, TRANSITION_AUTO[p.level], [])
            dur = {"cut": 0.14, "fade": 0.7 if p.level == "quiet" else 0.45}.get(style, 0.5)
            p.transition_in, p.transition_s = style, min(dur, 0.35 * (p.end - p.start))
            p.fade_in = False
            plans[i - 1].fade_out = False
    # --- instrumental breaks / intro / outro-side gaps: scenes only, never invented lyrics
    inter = []
    if cfg.interludes and music is not None and music.reliable:
        bounds = [(0.0 if not events else music.offset, events[0].start if events else 0.0)]
        for a, b in zip(events, events[1:]):
            bounds.append((a.end, b.start))
        if events and total_end:
            bounds.append((events[-1].end, total_end))
        for k, (a, b) in enumerate(bounds):
            if b - a >= 2.5 and not music.is_quiet_gap(a + 0.3, b - 0.3):
                lvl = music.level_at((a + b) / 2, window=min(3.0, b - a))
                rnd = random.Random(f"{cfg.seed}:i{k}")
                scene = _pick(rnd, {"quiet": ["night", "dream", "galaxy"], "mid": ["stars", "galaxy", "dream"],
                                    "high": ["streaks", "galaxy", "city"]}[lvl], recent_scenes[-2:])
                inter.append(Interlude(a + 0.15, b - 0.15, scene, lvl))
    return Direction(plans, inter, notes, sem_src)
