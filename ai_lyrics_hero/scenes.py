"""Procedural animated background scenes chosen by the meaning of each lyric line.

Everything is drawn by code (no downloads). Scene choice is an EXACT word match against scenes.json;
no match -> plain night sky; a line that repeats an earlier line -> light streaks."""
import json
import math
import os
import random
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .animation import clamp
from .text_utils import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
PRIORITY = ["heart_fire", "eye", "moon", "rain", "romance", "city", "road", "nature", "galaxy", "stars", "dream", "night"]
SCENE_KINDS = PRIORITY + ["streaks", "sky"]


class SceneLexicon:
    def __init__(self, extra_path=None):
        data = json.load(open(os.path.join(HERE, "scenes.json"), encoding="utf-8"))
        self.scenes = {k: {normalize(f) for f in v} for k, v in data["scenes"].items()}
        self.ambiguous = {normalize(a) for a in data.get("ambiguous", [])}
        self.block = {normalize(b) for b in data.get("block", [])}
        if extra_path:
            ex = json.load(open(extra_path, encoding="utf-8"))
            for k, v in ex.get("scenes", {}).items():
                self.scenes.setdefault(k, set()).update(normalize(f) for f in v)
            self.ambiguous |= {normalize(a) for a in ex.get("ambiguous", [])}
            self.block |= {normalize(b) for b in ex.get("block", [])}
            self.ambiguous -= {normalize(a) for a in ex.get("unambiguous", [])}

    def match(self, tokens, allow_ambiguous=False):
        """Return (scene_kind or None, matched_word or None, skipped_ambiguous_words)."""
        skipped, found = [], []
        order = PRIORITY + [k for k in self.scenes if k not in PRIORITY]
        for pos, tok in enumerate(tokens):
            w = normalize(tok)
            if not w or w in self.block:
                continue
            for kind in order:
                if w in self.scenes.get(kind, ()):
                    if w in self.ambiguous and not allow_ambiguous:
                        skipped.append(w)
                    else:
                        found.append((order.index(kind), pos, kind, w))
                    break
        if found:
            _, _, kind, w = min(found)            # strongest scene in the line; ties -> earliest word
            return kind, w, skipped
        return None, None, skipped


@dataclass
class ScenePlan:
    kind: str
    word: str
    fade_in: bool
    fade_out: bool


def plan_scenes(events, lexicon, cfg):
    """Returns (plans list aligned with events, notes). Same kind on consecutive lines keeps continuity."""
    plans, notes, seen = [], [], set()
    for i, ev in enumerate(events):
        kind, word, skipped = lexicon.match([w.text for w in ev.words], cfg.allow_ambiguous)
        if skipped:
            notes.append(f"line {i}: no scene for ambiguous word(s) {sorted(set(skipped))}")
        key = " ".join(normalize(w.text) for w in ev.words)
        toks = key.split()
        inner = len(toks) >= 4 and len(toks) % 2 == 0 and toks[:len(toks) // 2] == toks[len(toks) // 2:]
        if kind is None:
            kind, word = ("streaks", "(repeated line)") if (key in seen or inner) else ("sky", "")
        seen.add(key)
        plans.append(ScenePlan(kind, word, True, True))
    for i, p in enumerate(plans):
        if i > 0 and plans[i - 1].kind == p.kind and events[i].start - events[i - 1].end < 0.15:
            p.fade_in = False
        if i + 1 < len(plans) and plans[i + 1].kind == p.kind and events[i + 1].start - events[i].end < 0.15:
            p.fade_out = False
    return plans, notes


# ------------------------------------------------------------------ drawing helpers
def _radial(size, color, peak, power=2.0):
    s = int(size)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    r = np.sqrt((xx - s / 2) ** 2 + (yy - s / 2) ** 2) / (s / 2)
    a = np.clip(1 - r, 0, 1) ** power * peak
    out = np.zeros((s, s, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = color
    out[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _paste_c(frame, sprite, cx, cy, alpha=1.0, scale=1.0):
    if alpha <= 0.003 or scale <= 0.01:
        return
    if abs(scale - 1) > 0.003:
        sprite = sprite.resize((max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale))), Image.BILINEAR)
    if alpha < 0.997:
        sprite = sprite.copy(); sprite.putalpha(sprite.getchannel("A").point(lambda v: int(v * alpha)))
    frame.paste(sprite, (int(cx - sprite.width / 2), int(cy - sprite.height / 2)), sprite)


def _heart_points(cx, cy, size, n=160):
    pts = []
    for k in range(n):
        t = 2 * math.pi * k / n
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * size / 34.0, cy - y * size / 34.0))
    return pts


class SceneRenderer:
    def __init__(self, W, H, seed=5):
        self.W, self.H = W, H
        rnd = random.Random(seed)
        self.static_base = self._make_base(rnd)
        self.stars = [(rnd.random() * W, rnd.random() * H * 0.9, rnd.uniform(0, 6.28), rnd.uniform(0.8, 2.2))
                      for _ in range(36)]
        self.dense = [(rnd.random() * W, rnd.random() * H * 0.85, rnd.uniform(0, 6.28), rnd.uniform(0.7, 2.4))
                      for _ in range(90)]
        self.rain = [(rnd.random() * W * 1.3, rnd.random() * H, rnd.uniform(0.8, 1.6), rnd.uniform(40, 110))
                     for _ in range(95)]
        self.streaks = [(rnd.random() * W, rnd.random() * H, rnd.uniform(160, 420), rnd.uniform(0, 6.28),
                         rnd.choice([-1, 1])) for _ in range(16)]
        self.sparks = [(rnd.random() * W, rnd.random() * H, rnd.uniform(30, 90), rnd.uniform(1.0, 2.8)) for _ in range(46)]
        self.bokeh = [(rnd.random() * W, rnd.random() * H * 0.8, rnd.uniform(0.04, 0.12) * W, rnd.uniform(0, 6.28))
                      for _ in range(16)]
        self._sprites = {}

    # ---------- base sky ----------
    def _make_base(self, rnd):
        W, H = self.W, self.H
        top, bot = np.array([13, 22, 58], np.float32), np.array([8, 13, 36], np.float32)
        k = (np.linspace(0, 1, H, dtype=np.float32) ** 0.9)[:, None, None]
        arr = np.repeat((top + (bot - top) * k), W, axis=1)
        img = Image.fromarray(arr.astype(np.uint8), "RGB")
        d = ImageDraw.Draw(img, "RGBA")
        step = max(40, W // 12)
        for x in range(0, W, step):
            d.line([(x, 0), (x, H)], fill=(120, 140, 255, 9), width=1)
        for y in range(0, H, step):
            d.line([(0, y), (W, y)], fill=(120, 140, 255, 9), width=1)
        for _ in range(170):
            x, y = rnd.random() * W, rnd.random() * H
            r = rnd.choice([0.7, 1.0, 1.3])
            a = rnd.randint(50, 170)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(215, 225, 255, a))
        return img

    def base(self, t, dense=0.0):
        """Twinkling sky. `dense` (0..1) fades in the extra star layer used by the moon/stars scenes."""
        f = self.static_base.copy()
        d = ImageDraw.Draw(f, "RGBA")
        for x, y, ph, r in self.stars:
            a = int(60 + 150 * (0.5 + 0.5 * math.sin(ph + t * 1.7)))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(235, 240, 255, a))
        if dense > 0.003:
            for x, y, ph, r in self.dense:
                a = int((60 + 150 * (0.5 + 0.5 * math.sin(ph + t * 1.7))) * dense)
                d.ellipse([x - r, y - r, x + r, y + r], fill=(235, 240, 255, a))
        return f

    # ---------- sprites (built once) ----------
    def _sprite(self, name):
        if name in self._sprites:
            return self._sprites[name]
        W = self.W
        if name == "moon":
            D = int(W * 0.36)
            glow = _radial(D * 2.7, (255, 244, 214), 0.42, 2.2)
            disc = Image.new("RGBA", (D * 2, D * 2), (0, 0, 0, 0))
            dd = ImageDraw.Draw(disc, "RGBA")
            dd.ellipse([4, 4, D * 2 - 4, D * 2 - 4], fill=(243, 237, 213, 255))
            cr = Image.new("RGBA", disc.size, (0, 0, 0, 0))
            dd = ImageDraw.Draw(cr)
            for cx, cy, r, a in [(0.32, 0.33, .13, 70), (0.55, 0.52, .17, 80), (0.40, 0.70, .08, 60), (0.70, 0.30, .06, 55),
                                 (0.66, 0.74, .10, 55), (0.22, 0.58, .05, 50)]:
                dd.ellipse([D * 2 * (cx - r), D * 2 * (cy - r), D * 2 * (cx + r), D * 2 * (cy + r)], fill=(206, 198, 172, int(a * 0.8)))
            disc.alpha_composite(cr.filter(ImageFilter.GaussianBlur(D * 0.012)))
            disc = disc.resize((D, D), Image.LANCZOS)
            g = glow.copy(); g.alpha_composite(disc, ((g.width - D) // 2, (g.height - D) // 2))
            self._sprites[name] = g
        elif name == "eye":
            EW, EH = int(W * 0.60), int(W * 0.20)
            S = 2
            img = Image.new("RGBA", (EW * S, EH * S), (0, 0, 0, 0))
            xs = np.linspace(-1, 1, 200)
            up = [(EW * S * (0.5 + 0.5 * x), EH * S * (0.5 - 0.46 * (1 - x * x) ** 0.85)) for x in xs]
            lo = [(EW * S * (0.5 + 0.5 * x), EH * S * (0.5 + 0.40 * (1 - x * x) ** 0.85)) for x in xs[::-1]]
            shape = up + lo
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).polygon(shape, fill=255)
            body = Image.new("RGBA", img.size, (232, 232, 224, 255))
            iris_r = int(EH * S * 0.44)
            ic = (EW * S // 2, EH * S // 2)
            ir = _radial(iris_r * 2, (120, 150, 255), 1.0, 0.45).resize((iris_r * 2, iris_r * 2))
            ring = Image.new("RGBA", (iris_r * 2, iris_r * 2), (0, 0, 0, 0))
            rd = ImageDraw.Draw(ring)
            rd.ellipse([0, 0, iris_r * 2 - 1, iris_r * 2 - 1], fill=(88, 104, 214, 255))
            rd.ellipse([iris_r * 0.18, iris_r * 0.18, iris_r * 1.82, iris_r * 1.82], fill=(122, 148, 250, 255))
            pr = iris_r * 0.46
            rd.ellipse([iris_r - pr, iris_r - pr, iris_r + pr, iris_r + pr], fill=(8, 8, 18, 255))
            rd.ellipse([iris_r * 0.50, iris_r * 0.46, iris_r * 0.72, iris_r * 0.68], fill=(255, 255, 255, 255))
            body.alpha_composite(ring, (ic[0] - iris_r, ic[1] - iris_r))
            img.paste(body, (0, 0), mask)
            ImageDraw.Draw(img).line(up, fill=(25, 32, 70, 255), width=5 * S // 2)
            img = img.resize((EW, EH), Image.LANCZOS)
            self._sprites[name] = img
        elif name == "heart":
            D = int(W * 0.34)
            S = 2
            canvas = Image.new("RGBA", (D * 2 * S, D * 2 * S), (0, 0, 0, 0))
            d = ImageDraw.Draw(canvas, "RGBA")
            pts = _heart_points(D * S, D * S, D * S * 1.15)
            d.polygon(pts, fill=(255, 104, 56, 255))
            inner = _heart_points(D * S, D * S - D * S * 0.04, D * S * 0.9)
            d.polygon(inner, fill=(255, 122, 70, 255))
            hl = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            ImageDraw.Draw(hl).ellipse([D * S * 0.55, D * S * 0.55, D * S * 0.95, D * S * 0.8], fill=(255, 190, 140, 120))
            canvas.alpha_composite(hl.filter(ImageFilter.GaussianBlur(D * S * 0.05)))
            self._sprites[name] = canvas.resize((D * 2, D * 2), Image.LANCZOS)
        elif name == "heart_glow":
            D = int(W * 0.34)
            base = Image.new("L", (D * 4, D * 4), 0)
            ImageDraw.Draw(base).polygon(_heart_points(D * 2, D * 2, D * 1.35), fill=255)
            blur = base.filter(ImageFilter.GaussianBlur(D * 0.28))
            out = Image.new("RGBA", base.size, (255, 110, 40, 0))
            out.putalpha(blur.point(lambda v: int(v * 0.85)))
            self._sprites[name] = out
        elif name == "city":
            rnd = random.Random(21)
            Hh = int(self.H * 0.34)
            lay = Image.new("RGBA", (W, Hh), (0, 0, 0, 0))
            d = ImageDraw.Draw(lay, "RGBA")
            x = -20
            wins = []
            while x < W:
                bw = rnd.randint(int(W * 0.06), int(W * 0.13)); bh = rnd.randint(int(Hh * 0.30), int(Hh * 0.95))
                shade = rnd.randint(12, 26)
                d.rectangle([x, Hh - bh, x + bw, Hh], fill=(shade, shade + 4, shade + 22, 255))
                for wy in range(Hh - bh + 14, Hh - 10, 26):
                    for wx in range(x + 8, x + bw - 10, 20):
                        if rnd.random() < 0.38:
                            wins.append((wx, wy + 0, rnd.uniform(0, 6.28)))
                x += bw + rnd.randint(2, 10)
            self._city_windows = wins
            glow = _radial(int(W * 1.4), (255, 150, 70), 0.30, 1.6)
            self._sprites["city_glow"] = glow
            self._sprites[name] = lay
        elif name == "sun":
            self._sprites[name] = _radial(int(W * 0.55), (255, 190, 110), 0.95, 0.9)
        elif name == "mountains":
            rnd = random.Random(33)
            lay = Image.new("RGBA", (int(W * 1.6), int(self.H * 0.30)), (0, 0, 0, 0))
            d = ImageDraw.Draw(lay, "RGBA")
            for k, (col, amp, base) in enumerate([((44, 36, 84, 255), 0.55, 0.62), ((28, 24, 62, 255), 0.65, 0.78), ((16, 14, 40, 255), 0.5, 0.92)]):
                pts = [(0, lay.height)]
                x = 0
                while x <= lay.width + 80:
                    pts.append((x, lay.height * (base - amp * 0.35 * rnd.random() ** 1.3 * (1 + 0.4 * math.sin(x * 0.004 + k)))))
                    x += rnd.randint(60, 140)
                pts.append((lay.width, lay.height))
                d.polygon(pts, fill=col)
            self._sprites[name] = lay
        elif name == "galaxy_arms":
            rnd = random.Random(55)
            pts = []
            for i in range(900):
                arm = i % 3
                r = rnd.random() ** 0.7
                ang = arm * 2.094 + r * 5.5 + rnd.gauss(0, 0.22)
                col = [(190, 160, 255), (120, 170, 255), (255, 170, 220)][arm]
                pts.append((r, ang, rnd.uniform(0.8, 2.6), col, rnd.randint(70, 220)))
            self._galaxy_pts = pts
            self._sprites[name] = _radial(int(W * 1.1), (150, 120, 255), 0.38, 1.3)
        elif name == "heart_small":
            S = 36
            im = Image.new("RGBA", (S * 2, S * 2), (0, 0, 0, 0))
            ImageDraw.Draw(im).polygon(_heart_points(S, S, S * 1.5), fill=(255, 120, 170, 230))
            self._sprites[name] = im.filter(ImageFilter.GaussianBlur(0.8))
        elif name == "dreamblob":
            self._sprites[name] = _radial(int(W * 0.7), (170, 140, 255), 0.5, 1.2)
        elif name == "blob":
            self._sprites[name] = _radial(int(W * 0.55), (255, 120, 40), 0.55, 1.6)
        elif name == "warm":
            self._sprites[name] = _radial(int(self.H * 0.9), (170, 55, 25), 0.8, 1.2)
        elif name.startswith("bokeh"):
            r = int(name.split(":")[1]); col = tuple(int(c) for c in name.split(":")[2].split(","))
            s = _radial(r * 2, col, 0.38, 0.6)
            ImageDraw.Draw(s).ellipse([0, 0, r * 2 - 1, r * 2 - 1], outline=col + (60,), width=2)
            self._sprites[name] = s
        return self._sprites[name]

    def _scaled(self, name, scale):
        q = round(scale * 40) / 40.0                         # quantize so frames reuse cached resizes
        key = (name, q)
        if key not in self._sprites:
            sp = self._sprite(name)
            self._sprites[key] = sp.resize((max(1, int(sp.width * q)), max(1, int(sp.height * q))), Image.BILINEAR)
        return self._sprites[key]

    # ---------- scene drawing (all take scene-relative time t) ----------
    def _bokeh(self, f, t, col, alpha=1.0, n=16, warm=False):
        for i, (x, y, r, ph) in enumerate(self.bokeh[:n]):
            c = col if not warm else [(255, 160, 90), (90, 200, 190), (255, 210, 140)][i % 3]
            spr = self._sprite(f"bokeh:{int(r)}:{c[0]},{c[1]},{c[2]}")
            _paste_c(f, spr, x + 18 * math.sin(t * 0.5 + ph), y + 14 * math.cos(t * 0.4 + ph), alpha * (0.6 + 0.4 * math.sin(t * 0.8 + ph)))

    def draw(self, f, kind, t):
        W, H = self.W, self.H
        d = ImageDraw.Draw(f, "RGBA")
        if kind == "moon":
            _paste_c(f, self._sprite("moon"), W / 2, H * 0.19 + 9 * math.sin(t * 0.9), 1.0, 1 + 0.015 * math.sin(t * 1.3))
        elif kind == "eye":
            blink = 1.0
            ph = t % 3.4
            if ph < 0.22:
                blink = abs(1 - ph / 0.11)
            blink = max(0.06, min(1.0, blink if ph < 0.22 else 1.0))
            e = self._sprite("eye")
            e = e.resize((e.width, max(2, int(e.height * blink))), Image.BILINEAR)
            self._bokeh(f, t, (140, 170, 255))
            _paste_c(f, e, W / 2, H * 0.2 + 6 * math.sin(t * 0.8))
        elif kind == "heart_fire":
            tint = Image.new("RGB", f.size, (58, 20, 16))      # warm dark red over the sky, BEFORE the art
            f.paste(Image.blend(f, tint, 0.55))
            _paste_c(f, self._sprite("warm"), W / 2, H * 0.22, 0.9)
            for i in range(5):
                a = 0.5 + 0.5 * math.sin(t * (1.1 + 0.3 * i) + i)
                _paste_c(f, self._scaled("blob", 0.9 + 0.3 * a), W * (0.25 + 0.14 * i) + 22 * math.sin(t * 0.7 + i),
                         H * 0.2 + 70 * math.cos(t * 0.9 + i * 1.3), 0.45 * a)
            beat = 1 + 0.045 * max(0, math.sin(t * 7.5)) ** 2 + 0.01 * math.sin(t * 2) + 0.05 * getattr(self, "_pulse", 0.0)
            _paste_c(f, self._scaled("heart_glow", beat), W / 2, H * 0.19, 0.9)
            _paste_c(f, self._scaled("heart", beat), W / 2, H * 0.19, 1.0)
            for x, y0, sp, r in self.sparks:
                y = (y0 - t * sp * 3) % (H * 0.75)
                a = int(160 * (0.5 + 0.5 * math.sin(t * 2 + x)))
                d.ellipse([x - r / 2, y - r / 2, x + r / 2, y + r / 2], fill=(255, 205, 100, a))
        elif kind == "rain":
            self._bokeh(f, t, (255, 170, 100), 0.9, 10, warm=True)
            for x, y0, sp, ln in self.rain:
                y = (y0 + t * 900 * sp) % (H + ln) - ln
                xx = x - 0.30 * (y0 + t * 900 * sp) % (W * 0.2)
                d.line([(xx, y), (xx - ln * 0.22, y + ln)], fill=(190, 210, 255, 70), width=2)
        elif kind == "streaks":
            for x, y, ln, ph, dr in self.streaks:
                off = (t * 40 * dr + ph * 30)
                x1, y1 = (x + off) % (W * 1.2), (y - off * 1.9) % H
                d.line([(x1, y1), (x1 + ln * 0.42, y1 - ln * 0.9)], fill=(215, 225, 255, 140), width=2)
        elif kind in ("night", "stars"):
            for k in range(5):
                pts = []
                for x in range(0, W + 30, 30):
                    pts.append((x, H * (0.56 + 0.025 * k) + 16 * math.sin(x * 0.012 + t * 0.9 + k * 1.1)))
                d.line(pts, fill=(150, 180, 255, 60), width=2)
        elif kind == "city":
            lay = self._sprite("city")
            _paste_c(f, lay, W / 2, H * 0.80, 1.0)
            _paste_c(f, self._sprites["city_glow"], W / 2, H * 0.70, 0.9)
            top = int(H * 0.80 - lay.height / 2)
            for wx, wy, ph in self._city_windows:
                if math.sin(ph + t * 0.9) > -0.35:
                    d.rectangle([wx, top + wy, wx + 8, top + wy + 12], fill=(255, 214, 140, int(150 + 80 * math.sin(ph + t))))
            self._bokeh(f, t, (255, 190, 120), 0.8, 9)
        elif kind == "road":
            hz = H * 0.46
            d.polygon([(W * 0.46, hz), (W * 0.54, hz), (W * 1.05, H), (-W * 0.05, H)], fill=(10, 12, 28, 235))
            _paste_c(f, _radial(int(W * 1.2), (255, 150, 90), 0.50, 1.6), W / 2, hz + 6, 0.9)
            for edge in (-1, 1):
                d.line([(W / 2 + edge * W * 0.04, hz), (W / 2 + edge * W * 0.55, H)], fill=(200, 210, 255, 150), width=3)
            for k in range(9):
                z = ((k / 9.0) + t * 0.35) % 1.0                 # 0 far .. 1 near
                y = hz + (H - hz) * z ** 2.2
                half = (W * 0.006 + W * 0.02 * z ** 2.2)
                ln = (H - hz) * (0.01 + 0.07 * z ** 2.2)
                d.polygon([(W / 2 - half, y), (W / 2 + half, y), (W / 2 + half * 1.2, y + ln), (W / 2 - half * 1.2, y + ln)],
                          fill=(255, 235, 170, int(60 + 180 * z)))
        elif kind == "nature":
            if "nature_sky" not in self._sprites:
                sky = Image.new("RGBA", (1, int(H * 0.74)))
                px = sky.load()
                for y in range(sky.height):
                    k = y / sky.height
                    px[0, y] = (int(35 + 220 * k ** 1.7), int(26 + 118 * k ** 1.6), int(84 - 14 * k), int(235 * k ** 0.8))
                self._sprites["nature_sky"] = sky.resize((W, sky.height))
            sky = self._sprites["nature_sky"]
            f.paste(sky, (0, 0), sky)
            if "nature_ground" not in self._sprites:
                g = Image.new("RGBA", (1, H - sky.height))
                gp = g.load()
                for y in range(g.height):
                    k = min(1.0, y / (g.height * 0.45))
                    gp[0, y] = (int(120 - 108 * k), int(62 - 52 * k), int(62 - 34 * k), 255)
                self._sprites["nature_ground"] = g.resize((W, g.height))
            f.paste(self._sprites["nature_ground"], (0, sky.height))
            _paste_c(f, self._sprite("sun"), W * 0.5, H * 0.66 + 10 * math.sin(t * 0.5), 1.0)
            m = self._sprite("mountains")
            off = int((t * 14) % (m.width - W))
            piece = m.crop((off, 0, off + W, m.height))
            f.paste(piece, (0, H - m.height), piece)
        elif kind == "galaxy":
            self._sprite("galaxy_arms")
            _paste_c(f, self._sprite("galaxy_arms"), W / 2, H * 0.30, 1.0)
            rot = t * 0.12
            R = W * 0.85
            for r, ang, sz, col, a in self._galaxy_pts if hasattr(self, "_galaxy_pts") else []:
                x = W / 2 + math.cos(ang + rot) * r * R
                y = H * 0.30 + math.sin(ang + rot) * r * R * 0.45
                d.ellipse([x - sz, y - sz, x + sz, y + sz], fill=col + (a,))
        elif kind == "dream":
            for i in range(6):
                _paste_c(f, self._sprite("dreamblob"), W * (0.2 + 0.15 * i) + 40 * math.sin(t * 0.3 + i * 1.7),
                         H * (0.2 + 0.1 * i) + 50 * math.cos(t * 0.25 + i), 0.45 + 0.2 * math.sin(t * 0.5 + i))
            self._bokeh(f, t, (200, 190, 255), 0.8, 12)
        elif kind == "romance":
            tint = Image.new("RGB", f.size, (70, 22, 74))
            f.paste(Image.blend(f, tint, 0.40))
            self._bokeh(f, t, (255, 150, 190), 0.8, 12)
            hs = self._sprite("heart_small")
            for i in range(14):
                x0 = (i * 83.0 + 37) % W
                y = (H * 0.95 - ((t * (28 + 9 * (i % 5)) + i * 140) % (H * 0.85)))
                _paste_c(f, hs, x0 + 24 * math.sin(t * 0.8 + i), y, 0.35 + 0.35 * ((i * 7) % 5) / 4, 0.5 + 0.12 * (i % 4))
        # 'stars' and 'sky' rely on the (denser) twinkling base

    def frame(self, t, plan, t_rel, alpha, asset=None, pulse=0.0):
        """Background for time t: base sky + the planned scene, with fade alpha.

        asset: optional user image (W x H RGB) that replaces the procedural art; pulse: 0..1 beat accent."""
        self._pulse = pulse
        dense = clamp(alpha) if (plan is not None and plan.kind in ("stars", "moon") and asset is None) else 0.0
        base = self.base(t, dense=dense)
        if plan is None or plan.kind in ("sky",) or alpha <= 0.003:
            return base
        if asset is not None:
            scene = asset.copy()
            if plan.kind in ("rain", "streaks"):
                self.draw(scene, plan.kind, t_rel)
            return Image.blend(base, scene, clamp(alpha)) if alpha < 0.997 else scene
        scene = base.copy()
        self.draw(scene, plan.kind, t_rel)
        return Image.blend(base, scene, clamp(alpha)) if alpha < 0.997 else scene


def apply_camera(img, zoom, pan_x=0.0, pan_y=0.0):
    """Camera-like zoom/pan on a finished frame (zoom >= 1; pan in -1..1 of the available slack)."""
    if zoom <= 1.0005:
        return img
    W, H = img.size
    w2, h2 = W / zoom, H / zoom
    cx = W / 2 + pan_x * (W - w2) / 2
    cy = H / 2 + pan_y * (H - h2) / 2
    box = (int(round(cx - w2 / 2)), int(round(cy - h2 / 2)), int(round(cx + w2 / 2)), int(round(cy + h2 / 2)))
    return img.crop(box).resize((W, H), Image.BILINEAR)
