"""Caption text style (reference look): small bold lines, the word being sung highlighted."""
from dataclasses import dataclass
from typing import List
from PIL import Image, ImageDraw
from .animation import clamp, ease_out_cubic
from .config import parse_color
from .fonts import get_font, script_of
from .layout import measure, line_height

WHITE = (246, 248, 255)


@dataclass
class CapWord:
    index: int
    text: str
    x: float
    y: float
    w: float


@dataclass
class CaptionLayout:
    words: List[CapWord]
    size: int
    script: str
    rect: tuple           # (x0, y0, x1, y1)
    line_h: float


def layout_caption(texts, cfg, W, H, center_y_frac):
    script = script_of(" ".join(texts))
    size = int(cfg.caption_size * W / 1080.0)
    max_w = W * 0.66
    while True:
        f = get_font("caption", script, size)
        space = size * 0.30
        widths = [measure(t, f)[0] for t in texts]
        lines, cur, cw = [], [], 0.0
        for i, w in enumerate(widths):
            add = w + (space if cur else 0)
            if cur and cw + add > max_w:
                lines.append(cur); cur, cw = [i], w
            else:
                cur.append(i); cw += add
        if cur:
            lines.append(cur)
        fits = all(sum(widths[i] for i in ln) + space * (len(ln) - 1) <= max_w + 1 for ln in lines)
        if (fits and len(lines) <= 5) or size <= 28:
            break
        size -= 3
    lh = line_height(f, size) * 1.12
    block_h = lh * len(lines)
    top = H * center_y_frac - block_h / 2
    top = max(H * 0.10, min(top, H * 0.90 - block_h))
    out = []
    for li, ln in enumerate(lines):
        total = sum(widths[i] for i in ln) + space * (len(ln) - 1)
        x = (W - total) / 2
        for i in ln:
            out.append(CapWord(i, texts[i], x, top + li * lh, widths[i]))
            x += widths[i] + space
    x0 = min(w.x for w in out); x1 = max(w.x + w.w for w in out)
    return CaptionLayout(out, size, script, (x0, top, x1, top + block_h), lh)


def active_word(event, t):
    """Index of the word being sung at time t (None before the first word)."""
    cur = None
    for i, w in enumerate(event.words):
        if w.start <= t:
            cur = i
    return cur


def draw_caption(frame, lay, event, t, cfg):
    rel, left = t - event.start, event.end - t
    a = clamp(rel / 0.25) * (clamp(left / max(0.05, cfg.exit_s + 0.05)) if cfg.exit_s > 0 else 1.0)
    if a <= 0.003:
        return
    rise = 18 * (1 - ease_out_cubic(clamp(rel / 0.3)))
    f = get_font("caption", lay.script, lay.size)
    hi = parse_color(cfg.caption_highlight, "white")
    cur = active_word(event, t)
    pad = 20
    x0, y0, x1, y1 = lay.rect
    layer = Image.new("RGBA", (int(x1 - x0) + 2 * pad, int(y1 - y0) + 2 * pad + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for w in lay.words:
        col = hi if w.index == cur else WHITE
        d.text((w.x - x0 + pad, w.y - y0 + pad), w.text, font=f, fill=col + (255,))
    if a < 0.997:
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * a)))
    frame.paste(layer, (int(x0 - pad), int(y0 - pad + rise)), layer)
