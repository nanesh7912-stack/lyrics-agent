"""Audio loading and visualizer levels. Missing/unreadable audio never aborts rendering."""
import subprocess
import numpy as np

SR = 16000


def load_audio(path, start=0.0, duration=None):
    """Decode to mono float32 at 16 kHz via ffmpeg. Returns None on failure."""
    if not path:
        return None
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, timeout=600).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return np.frombuffer(out, dtype=np.float32) if out else None


class Levels:
    """Band energies per time, smoothed, for an animated bar visualizer."""

    def __init__(self, samples, bars=28, offset=0.0):
        self.bars = bars
        self.offset = offset
        self.samples = samples
        self._prev = np.zeros(bars)

    def at(self, t):
        if self.samples is None or len(self.samples) < 1024:
            # idle shimmer so the visualizer never looks dead when audio is unavailable
            return 0.08 + 0.04 * np.abs(np.sin(np.arange(self.bars) * 0.7 + t * 2))
        i = max(0, int((t - self.offset) * SR))
        seg = self.samples[i:i + 1024]
        if len(seg) < 1024:
            seg = np.pad(seg, (0, 1024 - len(seg)))
        spec = np.abs(np.fft.rfft(seg * np.hanning(1024)))[1:257]
        bins = np.array_split(spec, self.bars)
        lv = np.array([b.mean() for b in bins])
        lv = np.log1p(lv * 6) / np.log1p(6 * 8)
        lv = np.clip(lv * np.linspace(1.4, 0.9, self.bars), 0, 1)
        self._prev = np.maximum(lv, self._prev * 0.72)
        return self._prev
