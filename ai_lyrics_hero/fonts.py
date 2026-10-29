"""Script-aware font selection with fallbacks. Never raises; falls back to PIL's default."""
import os
import unicodedata
from functools import lru_cache
from PIL import ImageFont

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")

LATIN_HERO = ["Poppins-BoldItalic.ttf", "Poppins-Bold.ttf"]
LATIN_CAPTION = ["Poppins-Bold.ttf"]
LATIN_SUPPORT = ["Lora-Italic-Variable.ttf", "Poppins-Bold.ttf"]
DEVA = ["FreeSansBold.ttf", "FreeSans.ttf"]
SYSTEM = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/mangalb.ttf", "C:/Windows/Fonts/nirmala.ttf",
]


def script_of(text):
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return "deva"
        if o > 0x2E80 or 0x0590 <= o <= 0x08FF:
            return "other"
    return "latin"


def _candidates(role, script):
    latin = {"hero": LATIN_HERO, "caption": LATIN_CAPTION}.get(role, LATIN_SUPPORT)
    names = DEVA if script == "deva" else latin
    paths = [os.path.join(ASSETS, n) for n in names]
    return paths + SYSTEM


@lru_cache(maxsize=256)
def get_font(role, script, size):
    size = max(8, int(size))
    for p in _candidates(role, script):
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                if role == "support" and script == "latin" and "Lora" in p:
                    try:
                        f.set_variation_by_name("Medium Italic")
                    except Exception:
                        pass
                return f
            except OSError:
                continue
    return ImageFont.load_default()


def has_case(text):
    return any(unicodedata.category(c) in ("Lu", "Ll") for c in text)
