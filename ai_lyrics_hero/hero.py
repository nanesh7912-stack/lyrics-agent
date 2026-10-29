"""Deterministic, conservative hero-word selection. No network, no randomness."""
import unicodedata
import zlib
from .config import AUTO_CYCLE, PALETTE, parse_color

STOPWORDS = {
    # English
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "is", "are", "was", "be", "i", "me", "my",
    "you", "your", "it", "its", "we", "he", "she", "they", "this", "that", "so", "do", "oh", "uh", "ah",
    # Romanized Hindi / Urdu
    "are", "re", "ae", "ay", "hai", "hain", "ho", "hoon", "hu", "tha", "thi", "to", "toh", "na", "ka", "ki",
    "ke", "ko", "se", "me", "mein", "main", "maine", "mujhe", "tu", "tum", "ye", "yeh", "woh", "wo", "koi",
    "kya", "aur", "bhi", "hi", "par", "pe", "ne", "ek", "jo", "kuch", "kahin",
    # Devanagari
    "à¤¹à¥ˆ", "à¤¹à¥ˆà¤‚", "à¤¤à¥‹", "à¤¨à¤¾", "à¤®à¥ˆà¤‚", "à¤®à¥ˆà¤‚à¤¨à¥‡", "à¤¨à¥‡", "à¤•à¥‹", "à¤¸à¥‡", "à¤®à¥‡à¤‚", "à¤•à¤¾", "à¤•à¥€", "à¤•à¥‡", "à¤¯à¥‡", "à¤µà¥‹", "à¤•à¥‹à¤ˆ",
    "à¤•à¥à¤¯à¤¾", "à¤”à¤°", "à¤­à¥€", "à¤¹à¥€", "à¤ªà¤°", "à¤ªà¥‡", "à¤à¤•", "à¤œà¥‹", "à¤°à¥‡", "à¤…à¤°à¥‡",
}


EMPHASIS = {normalize_w for normalize_w in (
    "love", "pyar", "pyaar", "ishq", "mohabbat", "dil", "jaan", "heart", "fire", "aag", "high", "forever", "never",
    "always", "tonight", "baby", "zindagi", "khwab", "sapna", "chaand", "raat", "aankhon", "aankhen", "neendein",
    "khushboo", "intezaar", "yaadein", "tanha", "dard", "pyaara", "sanam", "dilbar")}


def normalize(token):
    t = unicodedata.normalize("NFC", token)
    t = "".join(ch for ch in t if not unicodedata.category(ch).startswith("P"))
    return t.lower().strip()


def _is_stop(tok):
    return normalize(tok) in STOPWORDS


def _extend_phrase(event, i, cfg):
    """Optionally widen a short hero word into a 2-word phrase (e.g. 'KYA HUA')."""
    if cfg.max_phrase_words < 2:
        return i, i
    w = event.words
    if len(normalize(w[i].text)) > 4:
        return i, i
    for j in (i - 1, i + 1):  # prefer the preceding word
        if 0 <= j < len(w):
            lo, hi = min(i, j), max(i, j)
            if sum(len(x.text) for x in w[lo:hi + 1]) + 1 <= 14:
                return lo, hi
    return i, i


def select_heroes(events, cfg, music=None):
    """Fill event.hero for each event. Returns the same list. Never raises on odd input.

    Scoring: word length, position in the phrase, sung duration, emotional-word list, a musical accent under the
    word (when `music` is given), a penalty for repeating recent heroes and for always emphasizing the same position."""
    recent, counts, last_pos = [], {}, None
    for ev in events:
        ev.hero = None
        if not cfg.enabled or not ev.words or ev.duration < cfg.min_event_s:
            continue
        if ev.explicit_hero:
            ev.hero = ev.explicit_hero
        else:
            norms = [normalize(w.text) for w in ev.words]
            if not any(norms):
                continue
            # Keep mixed sung vocalizations together as one visual phrase.
            vocal_tokens = {
                "aa", "aaa", "ah", "aah", "ha", "haa",
                "oh", "ooh", "uh", "hmm", "hm",
                "la", "na", "da", "ra", "ta", "ya",
                "tu", "ru", "du", "ri", "di"
            }
            is_vocalization = (
                3 <= len(ev.words) <= 16
                and all(nm in vocal_tokens or not nm for nm in norms)
                and sum(bool(nm) for nm in norms) >= 3
            )

            if is_vocalization:
                ev.hero = (0, len(ev.words) - 1)
            elif len(ev.words) == 1:
                ev.hero = (0, 0)
            elif len(set(norms)) == 1 and len(ev.words) <= 6:
                ev.hero = (0, len(ev.words) - 1)          # chant such as "la la la"
            else:
                n = len(ev.words)
                best, best_score = None, 2.5
                for i, w in enumerate(ev.words):
                    nm = norms[i]
                    if not nm or _is_stop(w.text):
                        continue
                    score = float(min(len(nm), 10))
                    score += 2.0 if i == n - 1 else (1.0 if i == n - 2 else 0.0)
                    if ev.timing_reliable and ev.duration > 0:
                        score += min(3.0, 6.0 * (w.end - w.start) / ev.duration)
                    if nm in EMPHASIS:
                        score += 3.5
                    if music is not None and getattr(music, "reliable", False):
                        hit = music.nearest_impact(w.start, 0.12, 0.12)
                        if hit:
                            score += 2.5 * hit[0]
                    pos = "last" if i == n - 1 else ("first" if i == 0 else "mid")
                    if pos == last_pos:
                        score -= 0.4
                    if nm in recent[-4:]:
                        score -= 5.0
                    score -= 1.5 * counts.get(nm, 0)
                    if score > best_score:        # strict > keeps the earliest on ties
                        best, best_score = i, score
                if best is None:
                    # no content word: the last word of a short phrase is a fair focal point
                    continue
                ev.hero = _extend_phrase(ev, best, cfg)
                last_pos = "last" if best == n - 1 else ("first" if best == 0 else "mid")
        if ev.hero:
            key = normalize(" ".join(w.text for w in ev.words[ev.hero[0]:ev.hero[1] + 1]))
            recent.append(key)
            counts[key] = counts.get(key, 0) + 1
    return events


def hero_color(event, cfg):
    """Deterministic accent color for an event (stable across runs and frames)."""
    spec = (cfg.highlight_color or "auto").lower()
    if spec != "auto":
        return parse_color(spec)
    if getattr(event, "accent", None) in PALETTE:
        return PALETTE[event.accent]
    lo, hi = event.hero
    key = normalize(" ".join(w.text for w in event.words[lo:hi + 1]))
    name = AUTO_CYCLE[zlib.crc32(key.encode("utf-8")) % len(AUTO_CYCLE)]
    return PALETTE[name]


