"""Parse ASS 'Dialogue' lines into LyricEvents.

Supports karaoke tags (\\k, \\kf, \\ko) as word timing, falls back to proportional
interpolation (timing_reliable=False) when no usable word timing exists.
Explicit hero words are marked with [[double brackets]] in the lyric text.
"""
import re
import os
from .models import LyricEvent, Word

_TAG = re.compile(r"\{[^}]*\}")
_K = re.compile(r"\\k[fo]?(\d+)", re.I)
_MARK = re.compile(r"\[\[(.+?)\]\]")


def parse_time(s):
    h, m, rest = s.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def interpolate_words(tokens, start, end):
    """Split [start,end] across tokens proportionally to their length."""
    weights = [max(1, len(t)) for t in tokens]
    total = float(sum(weights))
    out, t = [], start
    for tok, w in zip(tokens, weights):
        d = (end - start) * w / total
        out.append(Word(tok, t, t + d))
        t += d
    return out


def _timing_ok(words, start, end):
    prev_end = start - 1e-3
    for w in words:
        if w.start < start - 1e-3 or w.start < prev_end - 1e-3 or w.end < w.start or w.end > end + 0.25:
            return False
        prev_end = w.end
    return True


def parse_text(text, start, end):
    """Return (words, explicit_range, reliable) for one ASS text field."""
    # The writer uses ASS escapes for literal backslashes and braces. Protect
    # them before interpreting karaoke tags, line breaks, and override blocks.
    markers = []
    for marker in ("\ue000", "\ue001", "\ue002"):
        while marker in text or marker in markers:
            marker = chr(ord(marker) + 1)
        markers.append(marker)
    escaped_slash, escaped_open, escaped_close = markers
    text = text.replace("\\\u2060", escaped_slash)
    text = text.replace(r"\{", escaped_open).replace(r"\}", escaped_close)
    text = text.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    # build segments: (centiseconds or None, plain text) from karaoke tags
    segs, pos, cur_k = [], 0, None
    for m in _TAG.finditer(text):
        if m.start() > pos:
            segs.append((cur_k, text[pos:m.start()]))
        k = _K.search(m.group(0))
        if k:
            cur_k = int(k.group(1))
        pos = m.end()
    if pos < len(text):
        segs.append((cur_k, text[pos:]))

    words, has_k, t = [], False, start
    explicit = []
    for k, seg in segs:
        toks = seg.split()
        if not toks:
            if k is not None:
                t += k / 100.0
                has_k = True
            continue
        if k is not None:
            has_k = True
            dur = k / 100.0
            sub = interpolate_words(toks, t, t + dur)
            t += dur
        else:
            sub = [Word(tk, 0, 0) for tk in toks]
        words.extend(sub)

    if not words:
        return [], None, False
    # explicit [[hero]] markers (may span several tokens)
    joined, idx_marks, open_idx = [], [], None
    for i, w in enumerate(words):
        tok = w.text
        if "[[" in tok and open_idx is None:
            open_idx = i
        if "]]" in tok and open_idx is not None:
            idx_marks.append((open_idx, i)); open_idx = None
        w.text = tok.replace("[[", "").replace("]]", "")
    words = [w for w in words if w.text]
    for w in words:
        w.text = w.text.replace(escaped_slash, "\\").replace(escaped_open, "{").replace(escaped_close, "}")
    explicit = idx_marks[0] if idx_marks else None
    if explicit and not (0 <= explicit[0] <= explicit[1] < len(words)):
        explicit = None

    reliable = has_k and all(w.end > w.start for w in words) and _timing_ok(words, start, end)
    if not reliable:
        words = interpolate_words([w.text for w in words], start, end)
    return words, explicit, reliable


def parse_ass(path_or_text):
    """Parse an ASS file path or raw ASS text. Returns events sorted by start."""
    raw = os.fsdecode(path_or_text)
    lead = raw.lstrip().lower()
    is_ass_text = "\n" in raw or lead.startswith(("[", "dialogue:", "format:"))
    if not is_ass_text:
        with open(raw, encoding="utf-8-sig") as f:
            raw = f.read()
    events, fmt = [], None
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("format:") and fmt is None and "text" in line.lower():
            fmt = [c.strip().lower() for c in line.split(":", 1)[1].split(",")]
        elif line.lower().startswith("dialogue:"):
            cols = fmt or ["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"]
            parts = line.split(":", 1)[1].split(",", len(cols) - 1)
            if len(parts) < len(cols):
                continue
            row = dict(zip(cols, parts))
            try:
                s, e = parse_time(row["start"]), parse_time(row["end"])
            except (ValueError, KeyError):
                continue
            if e <= s:
                continue
            words, explicit, reliable = parse_text(row["text"], s, e)
            if words:
                events.append(LyricEvent(s, e, words, explicit, reliable))
    events.sort(key=lambda ev: ev.start)
    return events
