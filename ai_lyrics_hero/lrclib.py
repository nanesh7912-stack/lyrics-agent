"""Best-effort exact-match lyric lookup through the public LRCLIB API.

Only track metadata and duration are sent to LRCLIB; audio files are never uploaded.
The returned lyrics are community-supplied and still need review before publishing.
"""
from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://lrclib.net/api/get"
TIMEOUT_SECONDS = 5
_rate_lock = threading.Lock()
_next_request_at = 0.0


def infer_track_info(audio_path, title=None, artist=None):
    """Use explicit UI/CLI values first; infer `Title - Artist` filenames as a fallback."""
    title = str(title or "").strip()
    artist = str(artist or "").strip()
    stem = Path(audio_path).stem
    if not title and not artist:
        parts = re.split(r"\s+-\s+", stem, maxsplit=1)
        if len(parts) == 2:
            title, artist = (part.strip() for part in parts)
        else:
            title = stem.strip()
    elif not title:
        title = stem.strip()
    elif not artist:
        parts = re.split(r"\s+-\s+", stem, maxsplit=1)
        if len(parts) == 2 and _normalize(parts[0]) == _normalize(title):
            artist = parts[1].strip()
    return title[:200], artist[:200]


def _normalize(value):
    folded = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(char for char in folded if char.isalnum())


def lookup_lyrics(audio_path, duration, title=None, artist=None):
    """Return `(lyrics, info, warning)`; gracefully falls back when lookup fails."""
    track_title, track_artist = infer_track_info(audio_path, title, artist)
    if not track_title or not track_artist:
        return None, None, "Add the song title and artist to try the lyric spelling lookup."

    params = {"track_name": track_title, "artist_name": track_artist}
    try:
        duration_value = float(duration)
    except (TypeError, ValueError, OverflowError):
        duration_value = None
    if duration_value is not None and 1 <= duration_value <= 3600:
        params["duration"] = str(int(round(duration_value)))
    request = Request(
        API_URL + "?" + urlencode(params),
        headers={"User-Agent": "AI-Lyrics-Hero/1.0 (https://github.com/nanesh7912-stack/lyrics-agent)"},
    )
    _wait_for_rate_limit()
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return None, None, f"Lyrics lookup returned HTTP {response.status}; using speech recognition."
            item = json.loads(response.read(1_000_000).decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return None, None, "No matching lyric entry was found; using speech recognition."
        if exc.code == 429:
            _remember_rate_limit(exc.headers.get("Retry-After", "1"))
            return None, None, "The lyrics service is rate-limited; using speech recognition."
        return None, None, f"Lyrics lookup returned HTTP {exc.code}; using speech recognition."
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return None, None, f"Lyrics lookup was unavailable ({type(exc).__name__}); using speech recognition."

    if not isinstance(item, dict):
        return None, None, "No matching lyric entry was found; using speech recognition."
    if _normalize(item.get("trackName")) != _normalize(track_title) or \
            _normalize(item.get("artistName")) != _normalize(track_artist):
        return None, None, "The lyric result did not exactly match the song title and artist; using speech recognition."
    try:
        returned_duration = float(item.get("duration"))
        requested_duration = duration_value
    except (TypeError, ValueError):
        returned_duration, requested_duration = None, None
    if returned_duration and requested_duration and abs(returned_duration - requested_duration) > 2.0:
        return None, None, "The lyric result appears to be a different recording; using speech recognition."

    word_synced = _word_synced_lrc(item.get("lyricsfile")) if item.get("hasWordSync") else None
    synced = item.get("syncedLyrics")
    plain = item.get("plainLyrics")
    lyrics = word_synced or (synced if isinstance(synced, str) and synced.strip() else plain)
    if not isinstance(lyrics, str) or not lyrics.strip():
        return None, None, "The matching entry has no usable lyrics; using speech recognition."
    kind = "word-synced" if word_synced else ("synced" if lyrics is synced else "plain")
    info = {
        "provider": "LRCLIB",
        "kind": kind,
        "track_name": item.get("trackName", track_title),
        "artist_name": item.get("artistName", track_artist),
        "album_name": item.get("albumName") or "",
        "duration": returned_duration,
        "has_word_sync": bool(item.get("hasWordSync")),
        "community_supplied": True,
    }
    return lyrics.strip(), info, None


def _word_synced_lrc(raw):
    """Convert LRCLIB Lyricsfile word starts to the project's supported word-LRC syntax."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        import yaml
        document = yaml.safe_load(raw)
    except Exception:
        return None
    lines = document.get("lines") if isinstance(document, dict) else None
    if not isinstance(lines, list) or not lines:
        return None

    def stamp(milliseconds):
        value = max(0, int(milliseconds))
        minutes, remainder = divmod(value, 60_000)
        seconds, millis = divmod(remainder, 1000)
        return f"{minutes:02d}:{seconds:02d}.{millis:03d}"

    rows = []
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("text"), str):
            continue
        text = line["text"].strip()
        if not text:
            continue
        words = line.get("words")
        valid_words = isinstance(words, list) and len(words) == len(text.split()) and bool(words)
        if valid_words:
            encoded_words = []
            for word in words:
                if not isinstance(word, dict) or not isinstance(word.get("text"), str):
                    valid_words = False
                    break
                try:
                    start_ms = int(word["start_ms"])
                    if start_ms < 0:
                        valid_words = False
                        break
                except (KeyError, TypeError, ValueError, OverflowError):
                    valid_words = False
                    break
                try:
                    end_ms = int(word.get("end_ms")) if word.get("end_ms") is not None else None
                except (TypeError, ValueError, OverflowError):
                    end_ms = None
                end_tag = f",{stamp(end_ms)}" if end_ms is not None and end_ms >= start_ms else ""
                encoded_words.append(f"<{stamp(start_ms)}{end_tag}>{word['text']}")
        try:
            line_start = int(line.get("start_ms", words[0]["start_ms"] if valid_words else 0))
        except (TypeError, ValueError, OverflowError):
            line_start = 0
        if line_start < 0:
            continue
        body = "".join(encoded_words) if valid_words else text
        rows.append(f"[{stamp(line_start)}]{body}")
    return "\n".join(rows) if rows else None


def _wait_for_rate_limit():
    global _next_request_at
    with _rate_lock:
        delay = max(0.0, _next_request_at - time.monotonic())
        if delay:
            time.sleep(delay)
        _next_request_at = time.monotonic() + 0.3


def _remember_rate_limit(retry_after):
    global _next_request_at
    try:
        delay = max(1.0, float(retry_after))
    except (TypeError, ValueError, OverflowError):
        delay = 1.0
    with _rate_lock:
        _next_request_at = max(_next_request_at, time.monotonic() + delay)
