"""Meaning-matched illustrations.

Safety rules (so a word never gets a picture of something else):
  * EXACT surface-form matching against a curated dictionary - no fuzzy/semantic guessing.
  * Words listed as `ambiguous` (e.g. Hindi 'sun' = listen, 'kal' = yesterday/tomorrow) get no image
    unless the user turns allow_ambiguous on.
  * Words listed as `block` never get an image (e.g. 'bar' in 'dil bar', 'kar' = do).
  * Multi-word phrases ('dil bar', 'blue eyes') are matched before single words.
  * No match -> no image. Nothing is ever downloaded.
"""
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .image_anim import choose_preset, PRESETS
from .text_utils import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
EMOJI_FONT = os.path.join(os.path.dirname(HERE), "assets", "fonts", "NotoColorEmoji.ttf")
EMOJI_FONT_FALLBACKS = [EMOJI_FONT, "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
                        "C:/Windows/Fonts/seguiemj.ttf"]


@dataclass
class Match:
    entry_id: str
    emoji: str
    first: int          # first word index in the event
    last: int           # last word index
    form: str
    phrase: bool = False


@dataclass
class Cue:
    entry_id: str
    emoji: str
    time: float          # absolute seconds: when the word is sung
    preset: str
    side: int            # -1 left, +1 right
    word: str


class Lexicon:
    def __init__(self, extra_path=None):
        data = json.load(open(os.path.join(HERE, "lexicon.json"), encoding="utf-8"))
        self.entries = {e["id"]: e for e in data["entries"]}
        self.phrases = list(data["phrases"])
        self.ambiguous = {normalize(a) for a in data["ambiguous"]}
        self.block = {normalize(b) for b in data["block"]}
        if extra_path:
            extra = json.load(open(extra_path, encoding="utf-8"))
            for e in extra.get("entries", []):
                self.entries[e["id"]] = e
            self.phrases += extra.get("phrases", [])
            self.ambiguous |= {normalize(a) for a in extra.get("ambiguous", [])}
            self.block |= {normalize(b) for b in extra.get("block", [])}
            self.ambiguous -= {normalize(a) for a in extra.get("unambiguous", [])}
        self.form_index = {}
        for e in self.entries.values():
            for f in e["forms"]:
                self.form_index.setdefault(normalize(f), e)
        self.phrase_index = {}
        for p in self.phrases:
            for f in p["forms"]:
                self.phrase_index[" ".join(normalize(x) for x in f.split())] = p

    def match(self, tokens, allow_ambiguous=False):
        """tokens: list[str] in order. Returns (matches, skipped_ambiguous)."""
        norm = [normalize(t) for t in tokens]
        used, matches, skipped = set(), [], []
        for n in (3, 2):                                   # phrases first, longest first
            for i in range(len(norm) - n + 1):
                if any(k in used for k in range(i, i + n)):
                    continue
                key = " ".join(norm[i:i + n])
                p = self.phrase_index.get(key)
                if p:
                    matches.append(Match(p["id"], p["emoji"], i, i + n - 1, key, True))
                    used.update(range(i, i + n))
        for i, w in enumerate(norm):                       # single-word fused form: 'dilbar'
            if i in used or not w:
                continue
            p = self.phrase_index.get(w)
            if p:
                matches.append(Match(p["id"], p["emoji"], i, i, w, True)); used.add(i); continue
            if w in self.block:
                continue
            e = self.form_index.get(w)
            if not e:
                continue
            if w in self.ambiguous and not allow_ambiguous:
                skipped.append(w); continue
            matches.append(Match(e["id"], e["emoji"], i, i, w))
            used.add(i)
        matches.sort(key=lambda m: m.first)
        return matches, skipped


def plan_cues(events, lexicon, cfg):
    """One illustration per lyric event at most; avoids repeating the same picture or preset.
    Returns (cues_by_event_index, review_notes)."""
    cues, notes = {}, []
    recent, prev_preset = [], None
    for idx, ev in enumerate(events):
        toks = [w.text for w in ev.words]
        ms, skipped = lexicon.match(toks, cfg.allow_ambiguous)
        if skipped:
            notes.append(f"line {idx}: no image for ambiguous word(s) {sorted(set(skipped))}")
        if not ms:
            continue
        window = recent[-2 * cfg.image_gap_events:] if cfg.image_gap_events else []
        fresh = [m for m in ms if m.entry_id not in window and m.emoji not in window]
        if not fresh:
            notes.append(f"line {idx}: image '{ms[0].entry_id}' skipped (shown recently)")
            continue
        hero_hit = [m for m in fresh if ev.hero and ev.hero[0] <= m.first <= ev.hero[1]]
        m = (hero_hit or fresh)[0]
        preset = choose_preset(m.entry_id, prev_preset, cfg.animation_intensity)
        side = 1 if (idx % 2 == 0) else -1
        cues[idx] = Cue(m.entry_id, m.emoji, ev.words[m.first].start, preset, side, m.form)
        recent.extend([m.entry_id, m.emoji])
        prev_preset = preset
    return cues, notes


def _emoji_sprite(ch, size):
    f = None
    for path in EMOJI_FONT_FALLBACKS:
        if not os.path.isfile(path):
            continue
        try:
            f = ImageFont.truetype(path, 109)
            break
        except OSError:
            continue
    if f is None:
        raise OSError("No color emoji font is installed; install Noto Color Emoji or provide an image asset.")
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((8, 8), ch, font=f, embedded_color=True)
    bb = im.getbbox()
    if not bb:
        raise ValueError(f"emoji {ch!r} not renderable")
    im = im.crop(bb)
    s = size / max(im.size)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)


@lru_cache(maxsize=128)
def load_sprite(entry_id, emoji, size, icons_dir=""):
    """User PNG (icons_dir/<id>.png, transparent) wins; otherwise the emoji glyph."""
    if icons_dir:
        p = os.path.join(icons_dir, entry_id + ".png")
        if os.path.exists(p):
            try:
                if os.path.getsize(p) <= 25 * 1024 * 1024:
                    with Image.open(p) as source:
                        if source.width * source.height <= 60_000_000:
                            im = source.convert("RGBA")
                        else:
                            im = None
                    if im is not None:
                        s = size / max(im.size)
                        return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
            except (OSError, ValueError, Image.DecompressionBombError):
                pass
    try:
        return _emoji_sprite(emoji, size)
    except (OSError, ValueError):
        return None


def soft_shadow(sprite, blur=14, opacity=110, offset=(0, 10)):
    pad = blur * 3
    sh = Image.new("RGBA", (sprite.width + 2 * pad, sprite.height + 2 * pad), (0, 0, 0, 0))
    a = Image.new("L", sh.size, 0)
    a.paste(sprite.getchannel("A"), (pad + offset[0], pad + offset[1]))
    a = a.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: v * opacity // 255)
    sh.putalpha(a)
    sh.alpha_composite(sprite, (pad, pad))
    return sh, pad
