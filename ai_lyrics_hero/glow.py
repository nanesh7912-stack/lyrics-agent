"""Soft, localized glow: gaussian aura + blurred text halo. No hard neon outline."""
import numpy as np
from PIL import Image, ImageFilter


def make_glow(text_alpha, color, pad=None):
    """text_alpha: 'L' image of the hero text. Returns RGBA glow (larger than the text)."""
    w, h = text_alpha.size
    pad = pad or int(max(w, h) * 0.45)
    W, H = w + 2 * pad, h + 2 * pad
    mask = Image.new("L", (W, H), 0)
    mask.paste(text_alpha, (pad, pad))
    r = max(8, int(h * 0.22))
    halo = np.asarray(mask.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) / 255.0
    halo = np.clip(halo * 1.7, 0, 1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx, sy = w * 0.40, h * 0.85
    aura = np.exp(-(((xx - W / 2) / sx) ** 2 + ((yy - H / 2) / sy) ** 2) * 0.5)
    a = np.clip(0.75 * halo + 0.42 * aura, 0, 1)
    # force a smooth zero at the sprite edge so no box is ever visible
    edge = np.clip(np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy]) / pad, 0, 1)
    a *= edge * edge * (3 - 2 * edge)   # smoothstep: exactly zero at the sprite border
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = color
    out[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA"), pad
