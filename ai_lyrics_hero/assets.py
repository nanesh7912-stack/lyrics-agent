"""User-provided background images: discovery, safe loading, aspect-aware cover crop, caching.

Directory layout (any number of roots via config.asset_dirs, os.pathsep separated):
    <root>/backgrounds/<scene>/*.png|jpg|jpeg|webp
No network access. Unreadable/oversized files are skipped and reported."""
import logging
import os
import zlib
from typing import Dict, List, Optional
from PIL import Image, ImageEnhance, ImageFilter

log = logging.getLogger("ai_lyrics_hero")
EXTS = (".png", ".jpg", ".jpeg", ".webp")
MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 60_000_000


class AssetLibrary:
    def __init__(self, dirs="", size=(1080, 1920)):
        self.size = size
        self.items: Dict[str, List[str]] = {}
        self.warnings: List[str] = []
        self.licenses: List[str] = []
        self._cache: Dict[str, Image.Image] = {}
        for root in [d for d in (dirs or "").split(os.pathsep) if d.strip()]:
            self._scan(root)

    def _scan(self, root):
        bg = os.path.join(root, "backgrounds")
        if not os.path.isdir(bg):
            self.warnings.append(f"asset dir '{root}' has no 'backgrounds' folder (ignored)")
            return
        for scene in sorted(os.listdir(bg)):
            sd = os.path.join(bg, scene)
            if not os.path.isdir(sd):
                continue
            for fn in sorted(os.listdir(sd)):
                path = os.path.join(sd, fn)
                if fn.lower().endswith(EXTS) and os.path.isfile(path):
                    if os.path.getsize(path) > MAX_BYTES:
                        self.warnings.append(f"skipped {path}: larger than 25 MB")
                        continue
                    self.items.setdefault(scene, []).append(path)
        lic = os.path.join(root, "LICENSES.txt")
        if os.path.isfile(lic):
            try:
                self.licenses.append(open(lic, encoding="utf-8", errors="replace").read(400).strip())
            except OSError:
                pass

    def has(self, scene):
        return bool(self.items.get(scene))

    def pick(self, scene, key, avoid=()):
        """Deterministic choice among this scene's images, avoiding recently used paths when possible."""
        paths = self.items.get(scene, [])
        if not paths:
            return None
        fresh = [p for p in paths if p not in avoid] or paths
        return fresh[zlib.crc32(key.encode("utf-8")) % len(fresh)]

    def load(self, path) -> Optional[Image.Image]:
        """Cover-cropped (to 1.15x the frame for panning), slightly darkened RGB image, or None if unusable."""
        if path in self._cache:
            return self._cache[path]
        W, H = self.size
        try:
            with Image.open(path) as im:
                if im.width * im.height > MAX_PIXELS:
                    raise ValueError("image has too many pixels")
                im = im.convert("RGB")
            tw, th = int(W * 1.15), int(H * 1.15)
            s = max(tw / im.width, th / im.height)                  # cover, never stretch
            im = im.resize((max(tw, int(im.width * s + .5)), max(th, int(im.height * s + .5))), Image.LANCZOS)
            l, t = (im.width - tw) // 2, (im.height - th) // 2
            im = im.crop((l, t, l + tw, t + th))
            im = ImageEnhance.Brightness(im).enhance(0.58)           # keep lyrics readable
            self._cache[path] = im
            return im
        except Exception as e:
            self.warnings.append(f"could not use {path}: {e}")
            log.warning("asset skipped (%s): %s", path, e)
            self._cache[path] = None
            return None

    def frame(self, path, progress, pan=(1, 0)):
        """The visible W x H window of the image with a slow Ken-Burns zoom/pan (progress 0..1)."""
        im = self.load(path)
        if im is None:
            return None
        W, H = self.size
        z = 1.0 + 0.06 * progress
        cw, ch = int(im.width / 1.15 / z * 1.0), int(im.height / 1.15 / z * 1.0)
        cx = im.width / 2 + pan[0] * (im.width - cw) * 0.5 * (progress - 0.5)
        cy = im.height / 2 + pan[1] * (im.height - ch) * 0.5 * (progress - 0.5)
        box = (int(cx - cw / 2), int(cy - ch / 2), int(cx - cw / 2) + cw, int(cy - ch / 2) + ch)
        return im.crop(box).resize((W, H), Image.BILINEAR)
