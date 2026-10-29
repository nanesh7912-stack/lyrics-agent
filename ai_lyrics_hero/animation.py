"""Hero-word animation timeline (pure functions; no rendering)."""
from dataclasses import dataclass
import math


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    return 4 * t ** 3 if t < .5 else 1 - (-2 * t + 2) ** 3 / 2


def back_out(t, s):
    t -= 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


@dataclass
class HeroState:
    support_alpha: float
    support_dy: float
    hero_alpha: float
    hero_scale: float
    glow: float          # 0..1 multiplier applied to configured glow intensity
    active: bool         # event on screen at all
    dx: float = 0.0      # px offset of the hero block
    dy: float = 0.0
    rot: float = 0.0     # degrees
    reveal: float = 1.0  # 0..1 fraction of the width revealed (wipe preset)
    blur: float = 0.0    # gaussian radius px (blur preset)
    tracking: float = 0.0  # <0 = letters squeezed together (tracking preset), 0 = normal spacing
    chars: float = 1.0   # 0..1 fraction of characters shown (chars preset)
    exit_dx: float = 0.0


def hero_start_offset(event, cfg):
    """Seconds after event start at which the hero word begins (its sung time).

    Falls back to a short fixed lead if timing is missing/unreliable so the hero never
    appears late or outside the event."""
    if not event.hero:
        return 0.0
    off = event.words[event.hero[0]].start - event.start   # parser interpolates when unreliable
    latest = max(0.0, event.duration - cfg.exit_s - 0.12)
    return clamp(off, 0.0, latest)


TEXT_PRESETS = ("pop", "slide_left", "slide_right", "slide_up", "slide_down", "rotate", "wipe", "tracking",
                "blur", "chars", "impact")


def hero_state(t, event, cfg, preset="pop"):
    """State at absolute time t for an event rendered in hero mode.

    `preset` selects the entrance style; all presets share the same timing, glow ramp and exit, and every
    effect settles exactly to the neutral state (scale 1, no offset/rotation/blur) at the end of the entrance."""
    rel = t - event.start
    dur = event.duration
    if rel < 0 or rel >= dur:
        return HeroState(0, 0, 0, 0, 0, False)
    inten = cfg.animation_intensity
    # supporting text: restrained fade + small upward drift
    s_in = clamp(rel / max(0.05, cfg.entrance_s * 0.8))
    support_alpha = ease_out_cubic(s_in)
    support_dy = 14.0 * inten * (1 - ease_out_cubic(s_in))
    # hero: scale-up with controlled overshoot
    h_rel = rel - hero_start_offset(event, cfg)
    p = clamp(h_rel / cfg.entrance_s)
    start_scale = 1.0 - 0.28 * inten
    extra = {}
    if h_rel < 0:
        hero_alpha, hero_scale, glow = 0.0, start_scale, 0.0
        if preset not in ("pop", "impact"):
            hero_scale = 1.0
    else:
        a_fast = clamp(h_rel / (cfg.entrance_s * 0.45))
        glow = ease_in_out(clamp((h_rel - cfg.entrance_s * 0.35) / (cfg.entrance_s * 1.3)))
        hero_alpha, hero_scale = a_fast, 1.0
        e = ease_out_cubic(p)
        if preset == "pop" or preset not in TEXT_PRESETS:
            overshoot = 1.2 * inten
            hero_scale = start_scale + (1 - start_scale) * back_out(p, overshoot) if inten > 0 else 1.0
        elif preset.startswith("slide"):
            dist = 520.0 * (0.6 + 0.4 * inten)
            k = 1 - (back_out(p, 0.8 * inten) if inten > 0 else e)
            ux, uy = {"slide_left": (-1, 0), "slide_right": (1, 0), "slide_up": (0, 1), "slide_down": (0, -1)}[preset]
            extra.update(dx=ux * dist * k, dy=uy * dist * 0.6 * k)
            hero_alpha = clamp(h_rel / (cfg.entrance_s * 0.35))
        elif preset == "rotate":
            hero_scale = (0.8 + 0.2 * back_out(p, 1.0 * inten)) if inten > 0 else 1.0
            extra.update(rot=-16.0 * inten * (1 - e))
        elif preset == "wipe":
            extra.update(reveal=e)
            hero_alpha = 1.0
        elif preset == "tracking":
            extra.update(tracking=-0.10 * inten * (1 - e))
            hero_scale = 0.94 + 0.06 * e
        elif preset == "blur":
            extra.update(blur=14.0 * inten * (1 - e))
            hero_scale = 1.0 + 0.06 * inten * (1 - e)
            hero_alpha = clamp(h_rel / (cfg.entrance_s * 0.6))
        elif preset == "chars":
            extra.update(chars=clamp(h_rel / (cfg.entrance_s * 1.2)))
            hero_alpha = 1.0
        elif preset == "impact":
            q = clamp(h_rel / (cfg.entrance_s * 0.55))
            hero_scale = 1.0 + 0.55 * inten * (1 - ease_out_cubic(q))
            shake = 16.0 * inten * (1 - q) ** 2 * math.sin(q * 38)
            extra.update(dx=shake, dy=-shake * 0.4)
            glow = max(glow, 1 - q)                                   # brief light flash on the hit
    # exit
    left = dur - rel
    if cfg.exit_s > 0 and left < cfg.exit_s:
        q = clamp(left / cfg.exit_s)          # 1 -> 0
        e = q * q * (3 - 2 * q)            # smoothstep: fully gone exactly at the event end
        support_alpha *= e
        hero_alpha *= e
        glow *= e
        hero_scale *= 1.0 + 0.04 * inten * (1 - q)
        if preset.startswith("slide") and abs(extra.get("dx", 0.0)) < 1e-6:
            extra["dx"] = (-30.0 if preset in ("slide_left", "slide_up") else 30.0) * (1 - q) * inten
    return HeroState(support_alpha, support_dy, hero_alpha, hero_scale, glow, True, **extra)
