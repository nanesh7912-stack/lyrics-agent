"""Entrance presets for illustrations (reusable, deterministic, pure functions).

Each preset maps time-since-cue -> ImageState(dx, dy, scale, rot_deg, alpha); dx/dy are offsets
from the final position in canvas pixels. After the entrance the image holds still."""
import math
import zlib
from dataclasses import dataclass
from .animation import clamp, ease_out_cubic, back_out


@dataclass
class ImageState:
    dx: float
    dy: float
    scale: float
    rot: float
    alpha: float


def _lerp(a, b, t):
    return a + (b - a) * t


def pop(p, W, H, side, k):
    s = _lerp(0.55, 1.0, back_out(clamp(p), 2.6 * k + 0.01))
    return ImageState(0, 0, s, _lerp(-10 * k, 0, ease_out_cubic(clamp(p))), clamp(p / 0.25))


def throw(p, W, H, side, k):
    e = back_out(clamp(p), 1.5 * k + 0.01)
    return ImageState(_lerp(side * W * 0.8, 0, e), _lerp(H * 0.12, 0, e), 1.0,
                      _lerp(side * 28 * k, 0, e), clamp(p / 0.15))


def slam(p, W, H, side, k):
    q = clamp(p)
    shake = 14 * k * math.sin(q * 40) * (1 - q) ** 2
    return ImageState(shake, -shake * 0.5, _lerp(1.0 + 0.9 * k, 1.0, ease_out_cubic(q) ** 0.8), 0, clamp(p / 0.1))


def slide(p, W, H, side, k):
    e = back_out(clamp(p), 0.9 * k + 0.01)
    return ImageState(0, _lerp(H * 0.25, 0, e), 1.0, 0, clamp(p / 0.2))


def spin(p, W, H, side, k):
    e = ease_out_cubic(clamp(p))
    return ImageState(0, 0, _lerp(0.25, 1.0, e), _lerp(side * 300 * k, 0, e), clamp(p / 0.2))


def stamp(p, W, H, side, k):
    q = clamp(p)
    s = 1.0 + 0.55 * k * (1 - q) ** 2 - 0.05 * math.sin(q * math.pi)
    return ImageState(0, 0, s, side * -6 * (1 - ease_out_cubic(q)) * k, clamp(p / 0.08))


PRESETS = {"pop": (pop, 0.45), "throw": (throw, 0.55), "slam": (slam, 0.30),
           "slide": (slide, 0.50), "spin": (spin, 0.55), "stamp": (stamp, 0.35)}
ORDER = list(PRESETS)


def choose_preset(key, previous, intensity=1.0):
    """Deterministic pick from the entry id, never equal to the previous preset."""
    if intensity <= 0:
        return "pop"
    i = zlib.crc32(key.encode("utf-8")) % len(ORDER)
    if ORDER[i] == previous:
        i = (i + 1) % len(ORDER)
    return ORDER[i]


def image_state(t_since_cue, preset, W, H, side=1, intensity=1.0):
    fn, dur = PRESETS[preset]
    if t_since_cue < 0:
        return ImageState(0, 0, 0.0, 0, 0.0)
    return fn(t_since_cue / dur, W, H, side, max(0.0, intensity))
