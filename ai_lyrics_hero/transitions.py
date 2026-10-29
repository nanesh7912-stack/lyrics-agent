"""Scene-to-scene transitions between two full frames. p runs 0 (outgoing) -> 1 (incoming)."""
import math
from PIL import Image
from .animation import clamp, ease_in_out

STYLES = ("fade", "zoom", "slide", "wipe", "cut")


def _scale_about_center(img, s):
    if abs(s - 1) < 1e-3:
        return img
    W, H = img.size
    w2, h2 = int(W / s), int(H / s)
    l, t = (W - w2) // 2, (H - h2) // 2
    return img.crop((l, t, l + w2, t + h2)).resize((W, H), Image.BILINEAR)


def _wipe_mask(size, p, softness=0.18):
    W, H = size
    band = int(W * softness)
    pos = int((W + band) * p) - band
    row = Image.new("L", (W, 1))
    px = row.load()
    for x in range(W):
        k = (pos + band - x) / max(1, band)                         # 0..1 across the soft edge, 1 behind it
        px[x, 0] = int(255 * clamp(k))
    return row.resize((W, H))


def blend(style, a, b, p):
    """Return the frame at progress p for the given transition style."""
    p = clamp(p)
    if p <= 0.0:
        return a
    if p >= 1.0:
        return b
    e = ease_in_out(p)
    if style == "cut":
        flash = max(0.0, 1 - p * 4) * 0.5                            # a brief light flash on the cut
        out = b
        return Image.blend(out, Image.new("RGB", b.size, (255, 255, 255)), flash) if flash > 0.01 else out
    if style == "zoom":
        return Image.blend(_scale_about_center(a, 1 + 0.25 * e), _scale_about_center(b, 0.88 + 0.12 * e), e)
    if style == "slide":
        W = a.width
        out = a.copy()
        out.paste(b, (int(W * (1 - e)), 0))
        return out
    if style == "wipe":                                              # soft light wipe left -> right
        m = _wipe_mask(a.size, e)
        out = Image.composite(b, a, m)
        glow = 1 - abs(2 * e - 1)
        if glow > 0.05:
            out = Image.blend(out, Image.new("RGB", a.size, (255, 244, 220)), 0.16 * glow)
        return out
    return Image.blend(a, b, e)                                      # fade
