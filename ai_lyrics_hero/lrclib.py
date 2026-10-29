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
from difflib import SequenceMatcher
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


def _metadata_match(item, title, artist, duration, *, relaxed=False):
    """Score an LRCLIB record while rejecting ambiguous/wrong recordings."""
    if not isinstance(item, dict):
        return None

    result_title = str(item.get("trackName") or "")
    result_artist = str(item.get("artistName") or "")
    title_key = _normalize(title)
    artist_key = _normalize(artist)
    result_title_key = _normalize(result_title)
    result_artist_key = _normalize(result_artist)
    if not title_key or not result_title_key or not result_artist_key:
        return None

    if not relaxed:
        if result_title_key != title_key or result_artist_key != artist_key:
            return None
        title_score = artist_score = 1.0
    else:
        title_score = SequenceMatcher(None, title_key, result_title_key).ratio()
        expected_artist = {
            "".join(char for char in token if char.isalnum())
            for token in re.findall(r"[^\W_]+", unicodedata.normalize("NFKD", artist).casefold())
        }
        returned_artist = {
            "".join(char for char in token if char.isalnum())
            for token in re.findall(r"[^\W_]+", unicodedata.normalize("NFKD", result_artist).casefold())
        }
        if not expected_artist:
            artist_score = 1.0
        elif expected_artist.issubset(returned_artist):
            # Accept duet credits such as “Udit Narayan, Alka Yagnik”.
            artist_score = 1.0
        else:
            artist_score = SequenceMatcher(None, artist_key, result_artist_key).ratio()
        if title_score < 0.86 or artist_score < 0.72:
            return None

    try:
        result_duration = float(item.get("duration"))
    except (TypeError, ValueError, OverflowError):
        result_duration = None
    duration_difference = None
    duration_score = 0.5
    if result_duration is not None and duration is not None:
        duration_difference = abs(result_duration - duration)
        if duration_difference > (12.0 if relaxed else 2.0):
            return None
        duration_score = max(0.0, 1.0 - duration_difference / 12.0)

    word_sync_bonus = 0.03 if item.get("hasWordSync") else 0.0
    score = 0.48 * title_score + 0.32 * artist_score + 0.20 * duration_score + word_sync_bonus
    return score, duration_difference


def _has_lyrics(item):
    return bool(
        item.get("plainLyrics")
        or item.get("syncedLyrics")
        or (item.get("hasWordSync") and item.get("lyricsfile"))
    )


def _search_lyrics(track_title, track_artist, duration):
    """Search LRCLIB when its exact-match endpoint misses a metadata variant."""
    search_url = API_URL.rsplit("/", 1)[0] + "/search"
    queries = [
        {"track_name": track_title, "artist_name": track_artist},
        {"track_name": track_title},
    ]
    if duration is not None and 1 <= duration <= 3600:
        for params in queries:
            params["duration"] = str(int(round(duration)))

    matches = []
    for params in queries:
        request = Request(
            search_url + "?" + urlencode(params),
            headers={"User-Agent": "AI-Lyrics-Hero/1.0 (https://github.com/nanesh7912-stack/lyrics-agent)"},
        )
        _wait_for_rate_limit()
        try:
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                if response.status != 200:
                    continue
                results = json.loads(response.read(1_000_000).decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429:
                _remember_rate_limit(exc.headers.get("Retry-After", "1"))
                return None, "The lyrics service is rate-limited; using speech recognition."
            continue
        except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            return None, f"Lyrics search was unavailable ({type(exc).__name__}); using speech recognition."

        if not isinstance(results, list):
            continue
        for item in results:
            quality = _metadata_match(item, track_title, track_artist, duration, relaxed=True)
            if quality is not None and _has_lyrics(item):
                matches.append((quality[0], item))

    if not matches:
        return None, None
    matches.sort(key=lambda pair: pair[0], reverse=True)
    return matches[0][1], None


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
    item = None
    lookup_warning = None
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            if response.status != 200:
                lookup_warning = f"Lyrics lookup returned HTTP {response.status}; using speech recognition."
            else:
                item = json.loads(response.read(1_000_000).decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 429:
            _remember_rate_limit(exc.headers.get("Retry-After", "1"))
            return None, None, "The lyrics service is rate-limited; using speech recognition."
        if exc.code != 404:
            return None, None, f"Lyrics lookup returned HTTP {exc.code}; using speech recognition."
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        return None, None, f"Lyrics lookup was unavailable ({type(exc).__name__}); using speech recognition."

    exact_match = _metadata_match(item, track_title, track_artist, duration_value)
    searched = False
    if exact_match is None or not _has_lyrics(item):
        item, search_warning = _search_lyrics(track_title, track_artist, duration_value)
        searched = item is not None
        if item is None:
            if search_warning:
                return None, None, search_warning
            if lookup_warning:
                return None, None, lookup_warning
            return None, None, "No close matching lyric entry was found; using speech recognition."

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
        "duration": item.get("duration"),
        "has_word_sync": bool(item.get("hasWordSync")),
        "community_supplied": True,
        "match_method": "search" if searched else "exact",
    }
    warning = (
        "A close title, artist, and duration match was used from LRCLIB; check that it is the same recording."
        if searched else None
    )
    return lyrics.strip(), info, warning


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
