"""Two-level typography layout: supporting lines + a dominant hero block.

All geometry is computed in canvas pixels. The result is guaranteed (and tested) to stay
inside the portrait safe area and clear of the title and visualizer zones.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
from PIL import Image, ImageDraw
from .fonts import get_font, script_of, has_case

SAFE_X = 0.06          # horizontal margin fraction
TITLE_ZONE_BOTTOM = 0.14   # fraction of H reserved for the title
VIZ_ZONE_TOP = 0.78        # fraction of H where the visualizer zone begins
HERO_CENTER = 0.47


@dataclass
class TextLine:
    text: str
    size: int
    script: str
    role: str                      # "hero" | "support"
    cx: float
    top: float
    w: float
    h: float

    @property
    def rect(self):
        return (self.cx - self.w / 2, self.top, self.cx + self.w / 2, self.top + self.h)


@dataclass
class Layout:
    mode: str                                   # "hero" | "plain"
    hero_lines: List[TextLine] = field(default_factory=list)
    above: List[TextLine] = field(default_factory=list)
    below: List[TextLine] = field(default_factory=list)
    dropped_support: bool = False

    def all_lines(self):
        return self.above + self.hero_lines + self.below


def measure(text, font):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    return r - l, b - t


def line_height(font, size):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    l, t, r, b = d.textbbox((0, 0), "Hgjpy\u0915\u094d\u0937", font=font)
    return max(b - t, size) * 1.04


def _fit_one_line(text, script, role, max_w, start, floor):
    size = start
    while size > floor:
        w, _ = measure(text, get_font(role, script, size))
        if w <= max_w:
            return size
        size -= 4
    return floor


def _wrap(words, script, role, size, max_w):
    lines, cur = [], []
    for w in words:
        trial = " ".join(cur + [w])
        if cur and measure(trial, get_font(role, script, size))[0] > max_w:
            lines.append(" ".join(cur))
            cur = []
        # A single unusually long token (or unbroken vocalization) must not exceed
        # the safe width. Split it by characters only as a last resort.
        if measure(w, get_font(role, script, size))[0] > max_w:
            # Preserve any words already accumulated on this line before the
            # oversized token is broken into character-sized chunks.
            if cur:
                lines.append(" ".join(cur))
                cur = []
            chunk = ""
            for ch in w:
                candidate = chunk + ch
                if chunk and measure(candidate, get_font(role, script, size))[0] > max_w:
                    lines.append(chunk)
                    chunk = ch
                else:
                    chunk = candidate
            cur = [chunk] if chunk else []
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _make_lines(texts, script, role, size, cx, top):
    out, y = [], top
    f = get_font(role, script, size)
    lh = line_height(f, size)
    for t in texts:
        w, _ = measure(t, f)
        out.append(TextLine(t, size, script, role, cx, y, w, lh))
        y += lh
    return out


def _fit_support(words, script, base, max_w, max_lines=2, floor=26):
    size = base
    while True:
        lines = _wrap(words, script, "support", size, max_w)
        if len(lines) <= max_lines and all(measure(l, get_font("support", script, size))[0] <= max_w for l in lines):
            return size, lines
        if size <= floor:
            return size, lines[:max_lines]
        size -= 3


def display_text(text, cfg):
    return text.upper() if cfg.uppercase and has_case(text) else text


def compute_layout(event, cfg, W, H, x_shift=0.0, hero_scale=1.0, support_scale=1.0):
    """Return a Layout for the event. Falls back to plain mode if no hero is set."""
    scale = W / 1080.0
    max_w = W * (1 - 2 * SAFE_X) - 2 * abs(x_shift)
    safe_top, safe_bot = H * TITLE_ZONE_BOTTOM + 40, H * VIZ_ZONE_TOP - 20
    cx = W / 2.0 + x_shift

    if not event.hero:
        words = [w.text for w in event.words]
        script = script_of(" ".join(words))
        size = int(64 * scale)
        lines = _wrap(words, script, "hero", size, max_w * 0.92)
        while len(lines) > 4 and size > 30:
            size -= 4; lines = _wrap(words, script, "hero", size, max_w * 0.92)
        f = get_font("hero", script, size)
        block = line_height(f, size) * len(lines)
        tl = _make_lines(lines, script, "hero", size, cx, H * HERO_CENTER - block / 2)
        return Layout("plain", hero_lines=tl)

    lo, hi = event.hero
    hero_words = [w.text for w in event.words[lo:hi + 1]]
    before = [w.text for w in event.words[:lo]]
    after = [w.text for w in event.words[hi + 1:]]
    htext = display_text(" ".join(hero_words), cfg)
    hscript = script_of(htext)
    target = int(cfg.hero_size * scale * hero_scale)

    size = _fit_one_line(htext, hscript, "hero", max_w, target, 40)
    hero_texts = [htext]
    if size < target * 0.55 and len(hero_words) > 1:   # long phrase: balanced two-line split
        best = None
        for k in range(1, len(hero_words)):
            a = display_text(" ".join(hero_words[:k]), cfg)
            b = display_text(" ".join(hero_words[k:]), cfg)
            s = min(_fit_one_line(a, hscript, "hero", max_w, target, 40),
                    _fit_one_line(b, hscript, "hero", max_w, target, 40))
            if best is None or s > best[0]:
                best = (s, [a, b])
        if best and best[0] > size:
            size, hero_texts = best

    f = get_font("hero", hscript, size)
    hero_h = line_height(f, size) * len(hero_texts)

    def build(drop_below):
        sscript = script_of(" ".join(before + after))
        ssize_base = int(cfg.support_size * scale * support_scale)
        gap = max(24, int(size * 0.18))
        above_lines = below_lines = []
        a_size = b_size = ssize_base
        if before:
            a_size, al = _fit_support(before, sscript, ssize_base, max_w)
            above_lines = al
        if after and not drop_below:
            b_size, bl = _fit_support(after, sscript, ssize_base, max_w)
            below_lines = bl
        ah = line_height(get_font("support", sscript, a_size), a_size) * len(above_lines) if above_lines else 0
        bh = line_height(get_font("support", sscript, b_size), b_size) * len(below_lines) if below_lines else 0
        total = ah + (gap if ah else 0) + hero_h + (gap if bh else 0) + bh
        return sscript, a_size, b_size, above_lines, below_lines, ah, bh, gap, total

    drop = False
    sscript, a_size, b_size, above_lines, below_lines, ah, bh, gap, total = build(False)
    avail = safe_bot - safe_top
    if total > avail:
        drop = bool(after)
        sscript, a_size, b_size, above_lines, below_lines, ah, bh, gap, total = build(True)
    if total > avail and above_lines:      # still too tall: keep only the last supporting line above
        above_lines = above_lines[-1:]
        ah = line_height(get_font("support", sscript, a_size), a_size)
        total = ah + gap + hero_h
    # Last-resort vertical fit: shrink hero typography before it can overlap the
    # title/visualizer zones. The normal path above remains unchanged.
    if total > avail:
        max_hero_h = max(1.0, avail - (ah + (gap if ah else 0) + (bh + gap if bh else 0)))
        size = max(24, int(size * min(1.0, max_hero_h / max(1.0, hero_h))))
        f = get_font("hero", hscript, size)
        hero_h = line_height(f, size) * len(hero_texts)
        total = ah + (gap if ah else 0) + hero_h + (gap if bh else 0) + bh
    stack_top = H * HERO_CENTER - total / 2
    stack_top = max(safe_top, min(stack_top, safe_bot - total))
    y = stack_top
    lay = Layout("hero", dropped_support=drop)
    if above_lines:
        lay.above = _make_lines(above_lines, sscript, "support", a_size, cx, y)
        y += ah + gap
    lay.hero_lines = _make_lines(hero_texts, hscript, "hero", size, cx, y)
    y += hero_h
    if below_lines:
        lay.below = _make_lines(below_lines, sscript, "support", b_size, cx, y + gap)
    return lay
