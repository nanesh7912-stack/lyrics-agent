"""Frame renderer: Video-A identity (title, geometric background, visualizer) + hero typography."""
import hashlib
import math
import random
from dataclasses import replace
from PIL import Image, ImageDraw, ImageFilter
from .animation import hero_state, clamp, hero_start_offset
from .fonts import get_font, script_of
from .glow import make_glow
from .hero import hero_color
from .layout import compute_layout, TITLE_ZONE_BOTTOM, VIZ_ZONE_TOP
from .image_anim import image_state
from .caption import layout_caption, draw_caption
from .scenes import SceneRenderer, apply_camera, night_palette
from .director import direct, Interlude
from .transitions import blend
from .layout import display_text
from .config import parse_color
from .imagery import Lexicon, plan_cues, load_sprite, soft_shadow
from .text_utils import normalize

WHITE = (255, 255, 255)


def _song_seed(cfg, events, title, music, explicit_seed):
    """Choose one stable visual seed per song, unless the user supplied a seed."""
    if cfg.seed != 0:
        return int(cfg.seed)
    if explicit_seed is not None:
        return int(explicit_seed)

    parts = [normalize(str(title or ""))]
    parts.extend(" ".join(normalize(str(w.text)) for w in ev.words) for ev in events)
    if music is not None:
        parts.append(f"duration={getattr(music, 'duration', 0.0):.2f}")
        parts.append(f"tempo={getattr(music, 'tempo', None)}")
        energy = getattr(music, "energy", None)
        if energy is not None and len(energy):
            # Keep the fingerprint small and stable while distinguishing songs
            # that have the same title and lyrics but different recordings.
            count = min(64, len(energy))
            if count == 1:
                samples = [energy[0]]
            else:
                samples = [energy[round(i * (len(energy) - 1) / (count - 1))] for i in range(count)]
            parts.append("energy=" + ",".join(f"{float(value):.3f}" for value in samples))

    payload = "\n".join(parts).encode("utf-8", errors="replace")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _text_sprite(text, font, fill, pad=24, tracking=0):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    img = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill=fill + (255,))
    return img, pad, (r - l, b - t)


def _paste_alpha(frame, sprite, x, y, alpha=1.0):
    if alpha <= 0.002:
        return
    if alpha < 0.998:
        a = sprite.getchannel("A").point(lambda v: int(v * alpha))
        sprite = sprite.copy(); sprite.putalpha(a)
    frame.paste(sprite, (int(round(x)), int(round(y))), sprite)


class FrameRenderer:
    def __init__(self, cfg, events, W=1080, H=1920, levels=None, title=None, seed=None, lexicon=None,
                 music=None, assets=None, llm=None, total_end=None):
        self.title = cfg.title if title is None else title
        self.seed = _song_seed(cfg, events, self.title, music, seed)
        # Use the resolved seed everywhere so backgrounds and line-level choices
        # stay consistent with each other and repeat for the same song.
        self.cfg = replace(cfg, seed=self.seed)
        self.events, self.W, self.H, self.levels = events, W, H, levels
        self.music, self.assets = music, assets
        self.cues, self.cue_notes = {}, []
        if self.cfg.images:
            self.lexicon = lexicon or Lexicon(self.cfg.lexicon_path or None)
            self.cues, self.cue_notes = plan_cues(events, self.lexicon, self.cfg)
        self._img = {}
        self._caps = {}
        self._variants = {}
        self.direction = direct(events, self.cfg, music, None, assets, llm, total_end)
        self.plans = self.direction.plans
        self.scene_r = SceneRenderer(W, H, seed=self.seed) if self.cfg.background == "scenes" else None
        self.scene_plans = self.plans if self.scene_r else []
        self.scene_notes = list(self.direction.notes)
        self._layouts, self._sprites = {}, {}
        self.base_full = self._background(1.0)
        self.base_dim = self._background(0.35)
        self._starts = [e.start for e in events]

    # ---------- background (Video A identity) ----------
    def _background(self, shape_strength):
        W, H = self.W, self.H
        top, bot, _ = night_palette(self.seed)
        grad = Image.new("RGB", (1, H))
        px = grad.load()
        for y in range(H):
            k = (y / H) ** 0.8
            px[0, y] = tuple(int(top[i] + (bot[i] - top[i]) * k) for i in range(3))
        img = grad.resize((W, H))
        rnd = random.Random(self.seed)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for i in range(7):                       # translucent circle chain, as in Video A
            r = int(W * (0.10 + 0.012 * i))
            cx, cy = int(W * (0.06 + 0.14 * i)), int(H * (0.12 + 0.08 * i))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, int(34 * shape_strength)))
        for i in range(3):                       # thin geometric lines
            y = int(H * (0.30 + 0.16 * i)) + rnd.randint(-20, 20)
            d.line([(0, y), (W, y - int(H * 0.05))], fill=(255, 255, 255, int(18 * shape_strength)), width=2)
        return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")

    def _draw_title(self, frame, alpha=1.0):
        if not self.title:
            return
        # Larger, responsive title; always fit within the portrait safe width.
        max_w = self.W * 0.86
        size = max(22, int(self.W * 0.054))
        script = script_of(self.title)
        title = self.title.upper() if script == "latin" else self.title
        while size > 18:
            f = get_font("hero", script, size)
            spr, _, _ = _text_sprite(title, f, WHITE, pad=8)
            if spr.width <= max_w:
                break
            size -= 2
        # Subtle shadow improves contrast without making the title heavy.
        shadow = spr.getchannel("A").filter(ImageFilter.GaussianBlur(max(1, int(self.W * 0.004))))
        shadow_img = Image.new("RGBA", spr.size, (10, 8, 24, 0))
        shadow_img.putalpha(shadow.point(lambda v: int(v * 0.62)))
        _paste_alpha(frame, shadow_img, (self.W - spr.width) / 2, self.H * 0.052 + 2, alpha)
        _paste_alpha(frame, spr, (self.W - spr.width) / 2, self.H * 0.052, alpha * 0.96)

    def _fit_hero_sprite(self, sprite, cx, cy):
        """Scale and position animated hero text inside title/visualizer safe zones."""
        margin_x = self.W * 0.06
        top = self.H * 0.145
        bottom = self.H * 0.775
        max_w = max(1, self.W - 2 * margin_x)
        max_h = max(1, bottom - top)
        factor = min(1.0, max_w / max(1, sprite.width), max_h / max(1, sprite.height))
        if factor < 0.999:
            sprite = sprite.resize((max(1, int(sprite.width * factor)), max(1, int(sprite.height * factor))), Image.Resampling.LANCZOS)
        x = max(margin_x, min(cx - sprite.width / 2, self.W - margin_x - sprite.width))
        y = max(top, min(cy - sprite.height / 2, bottom - sprite.height))
        return sprite, x, y

    def _draw_visualizer(self, frame, t, accent):
        if self.levels is None:
            return
        lv = self.levels.at(t)
        n = len(lv)
        W, H = self.W, self.H
        bw, gap = W * 0.016, W * 0.011
        total = n * bw + (n - 1) * gap
        x0, base = (W - total) / 2, H * 0.915
        d = ImageDraw.Draw(frame)
        maxh = H * 0.075
        for i, v in enumerate(lv):
            h = max(6, v * maxh)
            x = x0 + i * (bw + gap)
            d.rounded_rectangle([x, base - h, x + bw, base], radius=bw / 2, fill=(70, 140, 255))

    # ---------- composition / layout / sprite caches ----------
    def _eff_cfg(self, idx):
        """Per-line animation settings: calmer in quiet passages, snappier and stronger in loud ones."""
        key = ("cfg", idx)
        if key not in self._variants:
            p = self.plans[idx]
            mult = {"quiet": 1.35, "mid": 1.0, "high": 0.85}[p.level]
            self._variants[key] = replace(self.cfg, entrance_s=self.cfg.entrance_s * mult,
                                          animation_intensity=min(2.0, self.cfg.animation_intensity * p.intensity))
        return self._variants[key]

    def _comp(self, idx):
        c = self.plans[idx].composition
        if not self.events[idx].hero and c in ("bg_word",):
            return "centered"
        return c

    def _x_shift(self, idx):
        return (1 if idx % 2 == 0 else -1) * 0.10 * self.W if self._comp(idx) == "offset" else 0.0

    def _side(self, idx):
        if self._comp(idx) == "offset":
            return -1 if self._x_shift(idx) > 0 else 1
        return self.cues[idx].side if idx in self.cues else 1

    def _layout(self, idx):
        if idx not in self._layouts:
            ev = self.events[idx]
            minimal = self._comp(idx) == "minimal"
            self._layouts[idx] = compute_layout(ev, self.cfg, self.W, self.H, x_shift=self._x_shift(idx),
                                                hero_scale=0.82 if minimal else 1.0, support_scale=0.9 if minimal else 1.0)
        return self._layouts[idx]

    def _hero_sprites(self, idx):
        if idx in self._sprites:
            return self._sprites[idx]
        ev, lay = self.events[idx], self._layout(idx)
        color = hero_color(ev, self.cfg) if ev.hero else WHITE
        fill = tuple(int(255 - (255 - c) * 0.18) for c in color)
        lines = lay.hero_lines
        pad = 24
        bx0 = min(l.rect[0] for l in lines) - pad
        by0 = min(l.rect[1] for l in lines) - pad
        bx1 = max(l.rect[2] for l in lines) + pad
        by1 = max(l.rect[3] for l in lines) + pad
        fill_used = fill if lay.mode == "hero" else WHITE
        block = Image.new("RGBA", (int(bx1 - bx0), int(by1 - by0)), (0, 0, 0, 0))
        for l in lines:
            f = get_font("hero", l.script, l.size)
            spr, p, _ = _text_sprite(l.text, f, fill_used, pad=0)
            block.alpha_composite(spr, (int(l.cx - spr.width / 2 - bx0), int(l.top - by0 + (l.h - spr.height) / 2)))
        glow = make_glow(block.getchannel("A"), color) if lay.mode == "hero" else None
        sup = []
        for l in lay.above + lay.below:
            f = get_font("support", l.script, l.size)
            spr, p, _ = _text_sprite(l.text, f, (236, 236, 244), pad=10)
            sup.append((spr, l.cx - spr.width / 2, l.top + (l.h - spr.height) / 2))
        out = dict(block=block, origin=(bx0, by0), glow=glow, support=sup, color=color, fill=fill_used, lines=lines)
        self._sprites[idx] = out
        return out

    def _render_block(self, idx, tracking, nchars_frac):
        """Hero block drawn character by character (tracking squeeze / typewriter). Latin script only."""
        sp = self._hero_sprites(idx)
        bx0, by0 = sp["origin"]
        base = sp["block"]
        out = Image.new("RGBA", base.size, (0, 0, 0, 0))
        total = sum(len(l.text) for l in sp["lines"])
        limit = max(1, int(math.ceil(total * nchars_frac)))
        shown = 0
        d = ImageDraw.Draw(out)
        for l in sp["lines"]:
            f = get_font("hero", l.script, l.size)
            widths = [f.getlength(ch) for ch in l.text]
            gap = tracking * l.size
            full = sum(widths) + gap * (len(widths) - 1)
            x = l.cx - bx0 - full / 2
            # Match the baseline/vertical placement used by _hero_sprites. Drawing
            # every character at the raw line top caused glyphs with ascenders,
            # descenders, and Devanagari marks to jump or clip during reveal.
            bbox = f.getbbox(l.text or " ")
            glyph_h = max(1, bbox[3] - bbox[1])
            y = l.top - by0 + (l.h - glyph_h) / 2 - bbox[1]
            for ch, w in zip(l.text, widths):
                if shown >= limit:
                    break
                d.text((x, y), ch, font=f, fill=sp["fill"] + (255,))
                x += w + gap
                shown += 1
        return out

    def _block_for(self, idx, st):
        """The hero block image for the current typography state (cached by quantized parameters)."""
        sp = self._hero_sprites(idx)
        block = sp["block"]
        latin = all(l.script == "latin" for l in sp["lines"])
        if latin and (abs(st.tracking) > 0.004 or st.chars < 0.999):
            key = (idx, "tc", round(st.tracking * 50), int(st.chars * 24))
            if key not in self._variants:
                self._variants[key] = self._render_block(idx, st.tracking, st.chars)
            block = self._variants[key]
        if st.reveal < 0.999:
            # Feather the reveal edge so it does not slice visibly through glyphs.
            w = max(1, min(block.width, int(block.width * st.reveal)))
            mask = Image.new("L", block.size, 0)
            md = ImageDraw.Draw(mask)
            feather = max(2, int(block.width * 0.035))
            hard_end = max(0, w - feather)
            if hard_end:
                md.rectangle((0, 0, hard_end, block.height), fill=255)
            fade_start = max(0, w - feather)
            fade_w = max(1, w - fade_start)
            for xx in range(fade_start, min(block.width, w)):
                value = int(255 * (xx - fade_start + 1) / fade_w)
                md.line((xx, 0, xx, block.height), fill=value)
            from PIL import ImageChops
            block = block.copy()
            block.putalpha(ImageChops.multiply(block.getchannel("A"), mask))
        if st.blur > 0.6:
            r = round(st.blur)
            key = (idx, "b", r, id(block) if st.reveal < 0.999 else 0)
            blurred = block.filter(ImageFilter.GaussianBlur(r))
            block = blurred
        return block

    def _bg_word(self, idx):
        key = (idx, "bgw")
        if key not in self._variants:
            ev = self.events[idx]
            lo, hi = ev.hero
            text = display_text(" ".join(w.text for w in ev.words[lo:hi + 1]), self.cfg)
            size = int((self.H * 0.15) if self.H > self.W else (self.H * 0.24))
            f = get_font("hero", "latin" if all(ord(c) < 0x900 for c in text) else "deva", size)
            color = self._hero_sprites(idx)["color"]
            spr, _, _ = _text_sprite(text, f, color, pad=10)
            a = spr.getchannel("A").point(lambda v: int(v * 0.13))
            spr.putalpha(a)
            self._variants[key] = spr
        return self._variants[key]

    def _image_geom(self, idx):
        """Sprite + resting center for this event's illustration, kept below the text and above the visualizer."""
        if idx in self._img:
            return self._img[idx]
        cue, lay = self.cues[idx], self._layout(idx)
        W, H = self.W, self.H
        text_bottom = max(l.rect[3] for l in lay.all_lines())
        top = text_bottom + 36
        bottom = H * VIZ_ZONE_TOP - 10
        size = min(int(self.cfg.image_size * W / 1080.0), int(bottom - top))
        geom = None
        if size >= 110:
            spr = load_sprite(cue.entry_id, cue.emoji, size, self.cfg.icons_dir)
            if spr is not None:
                spr, pad = soft_shadow(spr)
                cx = W / 2 + self._side(idx) * W * 0.16
                cy = top + size / 2 + max(0, (bottom - top - size) * 0.35)
                geom = (spr, cx, cy)
        self._img[idx] = geom
        return geom

    def _draw_image(self, frame, t, idx, ev):
        cue = self.cues.get(idx)
        if not cue:
            return
        geom = self._image_geom(idx)
        if not geom:
            return
        spr, cx, cy = geom
        st = image_state(t - cue.time, cue.preset, self.W, self.H, cue.side, self.cfg.animation_intensity)
        left = ev.end - t
        a = st.alpha
        if self.cfg.exit_s > 0 and left < self.cfg.exit_s:
            q = clamp(left / self.cfg.exit_s)
            a *= q * q * (3 - 2 * q)
        if a <= 0.002 or st.scale <= 0.01:
            return
        im = spr
        if abs(st.scale - 1) > 0.004:
            im = im.resize((max(1, int(im.width * st.scale)), max(1, int(im.height * st.scale))), Image.LANCZOS)
        if abs(st.rot) > 0.3:
            im = im.rotate(st.rot, resample=Image.BICUBIC, expand=True)
        _paste_alpha(frame, im, cx + st.dx - im.width / 2, cy + st.dy - im.height / 2, a)

    def _active_index(self, t):
        for i, e in enumerate(self.events):
            if e.start <= t < e.end:
                return i
        return None

    # ---------- scenes ----------
    def _scene_index(self, t):
        for i, p in enumerate(self.plans):
            if p.start <= t < p.vis_end:
                return i
        return None

    def _interlude_at(self, t):
        for it in self.direction.interludes:
            if it.start <= t < it.end:
                return it
        return None

    def _pulse(self, t):
        m = self.music
        if m is None or not m.reliable or self.cfg.beat_intensity <= 0:
            return 0.0
        b = m.last_beat_before(t)
        return math.exp(-(t - b) / 0.15) * min(1.0, self.cfg.beat_intensity) if b is not None else 0.0

    def _render_scene(self, plan, t, alpha):
        rel = t - plan.start
        dur = max(0.01, plan.vis_end - plan.start)
        prog = clamp(rel / dur)
        asset = None
        if plan.asset_path and self.assets is not None:
            asset = self.assets.frame(plan.asset_path, prog, plan.pan)
        fr = self.scene_r.frame(t, plan, rel, alpha, asset=asset, pulse=self._pulse(t))
        if asset is None:
            zoom = plan.zoom0 + (plan.zoom1 - plan.zoom0) * prog
            fr = apply_camera(fr, zoom, plan.pan[0] * (prog - 0.5), plan.pan[1] * (prog - 0.5))
        return fr

    def _scene_frame(self, t):
        si = self._scene_index(t)
        if si is None:
            it = self._interlude_at(t)
            if it is not None:
                a = clamp((t - it.start) / 0.5) * clamp((it.end - t) / 0.5)
                return self.scene_r.frame(t, it, t - it.start, a, pulse=self._pulse(t))
            return self.scene_r.frame(t, None, 0.0, 0.0)
        p = self.plans[si]
        a = 1.0
        if p.fade_in:
            a *= clamp((t - p.start) / 0.35)
        if p.fade_out:
            a *= clamp((p.vis_end - t) / 0.30)
        fr = self._render_scene(p, t, a)
        if si > 0 and p.transition_in != "none" and t - p.start < p.transition_s:
            prev = self.plans[si - 1]
            fr = blend(p.transition_in, self._render_scene(prev, t, 1.0), fr, (t - p.start) / p.transition_s)
        return fr

    # ---------- music-driven accents ----------
    def _beat(self, t, ev, idx):
        """(scale_add, dx, dy): a short pulse (and, on the strongest hits, a tiny shake) on musical impacts."""
        m, bi = self.music, self.cfg.beat_intensity
        if m is None or not m.reliable or bi <= 0 or not ev.hero:
            return 0.0, 0.0, 0.0
        eff = self._eff_cfg(idx)
        start = ev.start + hero_start_offset(ev, eff) + eff.entrance_s
        if t < start or t > ev.end - eff.exit_s:
            return 0.0, 0.0, 0.0
        hit = m.nearest_impact(t, before=0.35, after=0.0)
        if not hit or hit[1] < start - 0.02:
            return 0.0, 0.0, 0.0
        s, ti = hit
        dt = t - ti
        pulse = 0.05 * bi * s * math.exp(-dt / 0.10)
        shake = 5.0 * bi * math.sin(dt * 70) * math.exp(-dt / 0.07) if (s > 0.8 and bi > 0.5) else 0.0
        return pulse, shake, -0.4 * shake

    # ---------- main entry ----------
    def render(self, t):
        idx = self._active_index(t)
        caption = self.cfg.text_style == "caption"
        st = None
        dim = 0.0
        ev = self.events[idx] if idx is not None else None
        if idx is not None and ev.hero and self.cfg.enabled and not caption:
            st = hero_state(t, ev, self._eff_cfg(idx), self.plans[idx].typography)
            dim = min(1.0, st.hero_alpha) * 0.75
        if self.scene_r is not None:
            frame = self._scene_frame(t)
        else:
            frame = Image.blend(self.base_full, self.base_dim, dim) if dim > 0.01 else self.base_full.copy()
        self._draw_title(frame)
        if self.cfg.visualizer:
            self._draw_visualizer(frame, t, None)
        if idx is None:
            return frame
        if caption:
            if idx not in self._caps:
                self._caps[idx] = layout_caption([w.text for w in ev.words], self.cfg, self.W, self.H, self._caption_y(idx))
            draw_caption(frame, self._caps[idx], ev, t, self.cfg)
            return frame
        sp = self._hero_sprites(idx)
        if st is None:                              # plain / fallback presentation
            rel, left = t - ev.start, ev.end - t
            a = clamp(rel / 0.2) * clamp(left / 0.15)
            bx, by = sp["origin"]
            _paste_alpha(frame, sp["block"], bx, by, a)
            self._draw_image(frame, t, idx, ev)
            return frame
        for spr, x, y in sp["support"]:
            _paste_alpha(frame, spr, x, y + st.support_dy, st.support_alpha)
        block, (bx, by) = sp["block"], sp["origin"]
        pulse, sdx, sdy = self._beat(t, ev, idx)
        s = st.hero_scale * (1 + pulse)
        cx, cy = bx + block.width / 2 + st.dx + sdx, by + block.height / 2 + st.dy + sdy
        if self._comp(idx) == "bg_word" and st.hero_alpha > 0.002:
            bw = self._bg_word(idx)
            drift = (t - ev.start) * (8 if self.H > self.W else 22)
            _paste_alpha(frame, bw, self.W / 2 - bw.width / 2 - drift + 60, self.H * 0.47 - bw.height / 2, st.hero_alpha)
        if s > 0.01 and st.hero_alpha > 0.002:
            if sp["glow"] is not None and st.glow > 0.01:
                g, gpad = sp["glow"]
                ga = g.getchannel("A").point(lambda v, k=st.glow * self.cfg.glow_intensity * 0.85: int(min(255, v * k)))
                gl = Image.new("RGBA", g.size, sp["color"] + (0,)); gl.putalpha(ga)
                if abs(s - 1) > 0.004:
                    gl = gl.resize((max(1, int(gl.width * s)), max(1, int(gl.height * s))), Image.BILINEAR)
                gl, gx, gy = self._fit_hero_sprite(gl, cx, cy)
                _paste_alpha(frame, gl, gx, gy, min(1.0, st.hero_alpha))
            spr = self._block_for(idx, st)
            if abs(s - 1) > 0.004:
                spr = spr.resize((max(1, int(spr.width * s)), max(1, int(spr.height * s))), Image.Resampling.LANCZOS)
            if abs(st.rot) > 0.3:
                spr = spr.rotate(st.rot, resample=Image.Resampling.BICUBIC, expand=True)
            spr, sx, sy = self._fit_hero_sprite(spr, cx, cy)
            _paste_alpha(frame, spr, sx, sy, st.hero_alpha)
        self._draw_image(frame, t, idx, ev)
        return frame

    def _caption_y(self, idx):
        if self.cfg.background != "scenes":
            return 0.62
        kind = self.plans[idx].scene
        if kind in ("moon", "eye", "heart_fire"):
            return 0.74
        if kind == "rain":
            return 0.58
        return 0.50 if idx % 2 == 0 else 0.78





