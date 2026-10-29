"""Parse user lyrics: plain text (one line per lyric line) or LRC (line or word timestamps)."""
import re
from dataclasses import dataclass, field
from typing import List, Optional

_LRC = re.compile(r"^\s*((?:\[\d+:\d+(?:[.:]\d+)?\])+)(.*)$")
_TS = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")
_WORD_TS = re.compile(r"<(\d+:\d+(?:[.:]\d+)?)(?:,(\d+:\d+(?:[.:]\d+)?))?>")


def _secs(m, s):
    return int(m) * 60 + float(s.replace(":", "."))


@dataclass
class LyricLine:
    text: str
    start: Optional[float] = None           # known only for LRC input
    word_starts: Optional[List[float]] = None
    word_ends: Optional[List[Optional[float]]] = None


def parse_lyrics(raw):
    """Return (lines, timed) where timed is True when every line carries an LRC timestamp."""
    lines = []
    for row in raw.replace("\r\n", "\n").split("\n"):
        row = row.strip()
        if not row or re.match(r"^\[(ti|ar|al|by|offset|length|re|ve):", row, re.I):
            continue
        m = _LRC.match(row)
        if m:
            starts = [_secs(a, b) for a, b in _TS.findall(m.group(1))]
            body = m.group(2).strip()
            wstarts = None
            wends = None
            if _WORD_TS.search(body):
                word_times = _WORD_TS.findall(body)
                wstarts = [
                    _secs(*start.split(":", 1)) if ":" in start else 0.0
                    for start, _end in word_times
                ]
                parsed_ends = [
                    _secs(*end.split(":", 1)) if end else None
                    for _start, end in word_times
                ]
                if any(end is not None for end in parsed_ends):
                    wends = parsed_ends
                body = _WORD_TS.sub(" ", body)
            body = " ".join(body.split())
            if body:
                for s in starts:
                    lines.append(LyricLine(body, s, wstarts, wends))
            continue
        if row.startswith("[") and row.endswith("]"):      # section marker like [Chorus]
            continue
        lines.append(LyricLine(" ".join(row.split())))
    timed = bool(lines) and all(l.start is not None for l in lines)
    if timed:
        lines.sort(key=lambda l: l.start)
    return lines, timed
