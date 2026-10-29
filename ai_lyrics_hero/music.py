"""Lightweight, deterministic music analysis (numpy only): energy, onsets, impacts, tempo/beat grid, sections.

Nothing here claims perfect beat tracking. `MusicMap.reliable` is False when the audio was missing/too short
or no stable pulse was found; callers then fall back to fixed, deterministic behavior."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

SR = 16000
FRAME, HOP = 1024, 320            # 20 ms hop at 16 kHz


@dataclass
class MusicMap:
    duration: float = 0.0
    offset: float = 0.0                       # absolute time of sample 0 (the clip start)
    tempo: Optional[float] = None
    beats: List[float] = field(default_factory=list)
    onsets: List[Tuple[float, float]] = field(default_factory=list)      # (time, strength 0..1)
    impacts: List[Tuple[float, float]] = field(default_factory=list)     # strong accents only
    energy_hop: float = 0.05
    energy: np.ndarray = field(default_factory=lambda: np.zeros(0))      # 0..1 per energy_hop
    quiet_below: float = 0.30
    high_above: float = 0.60
    reliable: bool = False
    notes: List[str] = field(default_factory=list)

    # --- queries (all take absolute times) ---
    def energy_at(self, t):
        if len(self.energy) == 0:
            return 0.5
        i = int((t - self.offset) / self.energy_hop)
        return float(self.energy[min(max(i, 0), len(self.energy) - 1)])

    def smooth_energy(self, t, window=1.5):
        if len(self.energy) == 0:
            return 0.5
        a = max(0, int((t - self.offset - window / 2) / self.energy_hop))
        b = max(a + 1, int((t - self.offset + window / 2) / self.energy_hop))
        return float(self.energy[a:b].mean()) if a < len(self.energy) else float(self.energy[-1])

    def level_at(self, t, window=1.5):
        if not self.reliable:
            return "mid"
        e = self.smooth_energy(t, window)
        return "quiet" if e < self.quiet_below else ("high" if e > self.high_above else "mid")

    def nearest_impact(self, t, before=0.0, after=0.0):
        """Strongest impact within [t-before, t+after] or None."""
        c = [(s, ti) for ti, s in self.impacts if t - before <= ti <= t + after]
        return max(c) if c else None

    def last_beat_before(self, t):
        if not self.beats:
            return None
        i = int(np.searchsorted(self.beats, t, side="right")) - 1
        return self.beats[i] if i >= 0 else None

    def is_quiet_gap(self, a, b, thresh=0.12):
        """True when the music is nearly silent between a and b."""
        if len(self.energy) == 0:
            return False
        i, j = int((a - self.offset) / self.energy_hop), int((b - self.offset) / self.energy_hop)
        seg = self.energy[max(0, i):max(i + 1, j)]
        return len(seg) > 0 and float(seg.max()) < thresh          # nothing at all happens in the window


def _spectral_flux(x):
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    mags = np.abs(np.fft.rfft(x[idx] * np.hanning(FRAME)[None, :], axis=1))
    logm = np.log1p(mags * 20)
    flux = np.maximum(0, np.diff(logm, axis=0)).sum(axis=1)
    rms = np.sqrt((x[idx] ** 2).mean(axis=1))
    return np.concatenate([[0], flux]), rms


def _pick_peaks(env, hop_s, min_gap=0.12):
    if len(env) < 5:
        return []
    k = max(3, int(0.5 / hop_s))
    pad = np.pad(env, (k, k), mode="edge")
    local = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(env))])
    mad = np.median(np.abs(env - np.median(env))) + 1e-9
    thr = local + 2.2 * mad
    peaks, last = [], -1e9
    for i in range(1, len(env) - 1):
        if env[i] > thr[i] and env[i] >= env[i - 1] and env[i] > env[i + 1] and i * hop_s - last >= min_gap:
            peaks.append(i); last = i * hop_s
    return peaks


def _tempo(env, hop_s):
    """Autocorrelation tempo in 60-180 BPM; returns (bpm, confidence) or (None, 0)."""
    if len(env) < int(4 / hop_s):
        return None, 0.0
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    lo, hi = int(60 / 180 / hop_s), int(60 / 60 / hop_s)
    seg = ac[lo:hi + 1]
    if len(seg) < 3 or ac[0] <= 0:
        return None, 0.0
    j = int(np.argmax(seg))
    conf = float(seg[j] / ac[0])
    lag = (lo + j) * hop_s
    return 60.0 / lag, conf


def analyze(samples, offset=0.0):
    """Analyze mono 16 kHz float samples. Never raises; returns an unreliable MusicMap on bad input."""
    try:
        if samples is None or len(samples) < SR * 2 or not np.isfinite(samples).all():
            m = MusicMap(duration=0.0 if samples is None else len(samples) / SR, offset=offset)
            m.notes.append("no usable audio for analysis; deterministic fallback in use")
            return m
        x = np.asarray(samples, dtype=np.float32)
        flux, rms = _spectral_flux(x)
        hop_s = HOP / SR
        dur = len(x) / SR
        # energy curve (0.05 s) normalized by its 95th percentile
        per = max(1, int(round(0.05 / hop_s)))
        eh = per * hop_s                                      # real hop of the energy curve (0.04 s here)
        n = len(rms) // per
        e = rms[:n * per].reshape(n, per).mean(axis=1)
        top = np.sort(e)[-max(1, int(len(e) * 0.35)):]
        ref = float(top.mean()) + 1e-9                    # typical level of the loud 35% of the song
        energy = np.clip(e / ref, 0, 1.5) / 1.5
        sm = np.convolve(energy, np.ones(5) / 5, mode="same")
        # onsets
        fl = flux / (np.percentile(flux, 99) + 1e-9)
        pk = _pick_peaks(fl, hop_s)
        onsets = [(float(i * hop_s + offset), float(min(1.0, fl[i]))) for i in pk]
        thr = np.percentile([s for _, s in onsets], 85) if len(onsets) >= 6 else 1.1
        impacts, last = [], -9
        for t, s in onsets:
            if s >= thr and s >= 0.45 and t - last >= 0.45:
                impacts.append((t, s)); last = t
        bpm, conf = _tempo(fl, hop_s)
        beats = []
        reliable_pulse = bpm is not None and conf > 0.12 and len(onsets) >= 8
        if reliable_pulse:
            period = 60.0 / bpm
            # phase: the grid offset that best matches onset times
            ot = np.array([t - offset for t, _ in onsets]); ow = np.array([s for _, s in onsets])
            phases = np.linspace(0, period, 24, endpoint=False)
            score = [(ow * np.exp(-0.5 * ((((ot - ph + period / 2) % period) - period / 2) / 0.04) ** 2)).sum() for ph in phases]
            ph = phases[int(np.argmax(score))]
            beats = [float(offset + ph + k * period) for k in range(int((dur - ph) / period) + 1)]
        m = MusicMap(duration=dur, offset=offset, tempo=float(bpm) if reliable_pulse else None, beats=beats,
                     onsets=onsets, impacts=impacts, energy_hop=eh, energy=sm,
                     quiet_below=0.30, high_above=0.60, reliable=True)
        if not reliable_pulse:
            m.notes.append("no stable pulse found (sparse/free-tempo audio); beat grid disabled, onsets still used")
        return m
    except Exception as exc:                                   # analysis must never break rendering
        m = MusicMap(offset=offset)
        m.notes.append(f"audio analysis failed ({exc}); deterministic fallback in use")
        return m
