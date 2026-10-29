"""Lyrics (+ audio) -> timed LyricEvents with a review report.

Spelling policy: the user's or matched catalog lyric text is NEVER altered. Audio only provides *timing*.
Three timing sources, in order of accuracy:
  1. Word-synced LRC timestamps             (exact word starts/ends when provided)
  2. Line-synced LRC + ASR word times       (catalog/user spelling, vocal timing anchors)
  3. Line-synced LRC timestamps             (exact line starts; words interpolated)
  4. ASR word timestamps (faster-whisper)  (words matched to the lyrics by a spelling-tolerant alignment)
  5. Energy/voice-activity estimate         (rough; no ASR needed; always flagged 'estimated')
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .audio import SR, load_audio
from .lyrics import LyricLine
from .models import LyricEvent, Word
from .text_utils import similarity, syllables

log = logging.getLogger("ai_lyrics_hero")


@dataclass
class LineReport:
    index: int
    text: str
    start: float
    end: float
    method: str            # lrc | asr | estimated
    confidence: float      # 0..1
    needs_review: bool
    notes: List[str] = field(default_factory=list)
    words: List[dict] = field(default_factory=list)            # per word: text, start, end, source, confidence
    uncertain_words: List[dict] = field(default_factory=list)  # words whose timing is estimated/low confidence


def _word_info(ws, sources, confs, uncertain_below=0.8):
    info = [dict(text=w.text, start=round(w.start, 3), end=round(w.end, 3), source=src, confidence=round(c, 2))
            for w, src, c in zip(ws, sources, confs)]
    unc = [dict(i=i, **d) for i, d in enumerate(info)
           if d["source"] in ("interpolated", "estimated")
           or d["source"].endswith("inferred-end") or d["confidence"] < uncertain_below]
    return info, unc


def _split_words(text):
    return text.split()


def _distribute(words, start, end):
    """Place words across [start,end] proportionally to syllable weight."""
    w = [max(1, syllables(t)) + 0.3 * len(t) / 5 for t in words]
    total, t, out = float(sum(w)), start, []
    for tok, wt in zip(words, w):
        d = (end - start) * wt / total
        out.append(Word(tok, t, t + d))
        t += d
    return out


# ------------------------------------------------------------------ 1. LRC
def align_from_lrc(lines: List[LyricLine], audio_duration=None, tail=3.0):
    events, reports = [], []
    for i, ln in enumerate(lines):
        nxt = lines[i + 1].start if i + 1 < len(lines) else None
        words_txt = _split_words(ln.text)
        if ln.word_starts and len(ln.word_starts) == len(words_txt):
            end = nxt if nxt else ln.word_starts[-1] + 1.0
            ws = []
            for k, tok in enumerate(words_txt):
                explicit_end = ln.word_ends[k] if ln.word_ends and k < len(ln.word_ends) else None
                e = explicit_end if explicit_end is not None else (
                    ln.word_starts[k + 1] if k + 1 < len(words_txt)
                    else min(end, ln.word_starts[-1] + 1.0)
                )
                ws.append(Word(tok, ln.word_starts[k], max(e, ln.word_starts[k] + 0.05)))
            ev_end = ws[-1].end
            has_all_ends = bool(ln.word_ends) and len(ln.word_ends) == len(words_txt) and all(
                value is not None for value in ln.word_ends
            )
            rel = has_all_ends
            conf = 1.0 if has_all_ends else 0.75
            note = [] if has_all_ends else ["word end times inferred where the lyric source provided starts only"]
            srcs = [
                "lrc-word" if ln.word_ends and k < len(ln.word_ends) and ln.word_ends[k] is not None
                else "lrc-word-inferred-end"
                for k in range(len(ws))
            ]
        else:
            est_len = 0.38 * sum(max(1, syllables(t)) for t in words_txt) + 0.2
            end = min(nxt if nxt else ln.start + est_len + tail, ln.start + max(est_len, 1.0) + 1.5)
            ws = _distribute(words_txt, ln.start, ln.start + min(est_len, max(0.5, end - ln.start)))
            rel, conf, note = False, 0.7, ["word times interpolated within the line"]
            ev_end = ws[-1].end
            srcs = ["lrc-line" if k == 0 else "interpolated" for k in range(len(ws))]
        hold_to = min(nxt, ev_end + 1.2) if nxt else ev_end + 1.2
        events.append(LyricEvent(ln.start, max(hold_to, ev_end), ws, None, rel))
        winfo, unc = _word_info(
            ws, srcs,
            [1.0 if x == "lrc-word" else 0.7 if x.endswith("inferred-end") else 0.5 for x in srcs],
        )
        reports.append(LineReport(i, ln.text, ln.start, events[-1].end, "lrc", conf, not rel, note, winfo, unc))
    return events, reports


# ------------------------------------------------------------------ 2. ASR
def _nw_align(lyr_words, asr_words, gap=-0.45, thresh=0.55):
    """Needleman-Wunsch between lyric words and ASR words. Returns {lyric_idx: (asr_idx, sim)}."""
    n, m = len(lyr_words), len(asr_words)
    sim = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            sim[i][j] = similarity(lyr_words[i], asr_words[j].text)
    S = np.zeros((n + 1, m + 1), dtype=np.float32)
    S[:, 0] = np.arange(n + 1) * gap
    S[0, :] = np.arange(m + 1) * gap
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    back[1:, 0], back[0, 1:] = 1, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = sim[i - 1][j - 1]
            diag = S[i - 1, j - 1] + (2 * s - 1.0 if s >= thresh else -0.9)
            up, left = S[i - 1, j] + gap, S[i, j - 1] + gap
            best = max(diag, up, left)
            S[i, j] = best
            back[i, j] = 0 if best == diag else (1 if best == up else 2)
    i, j, pairs = n, m, {}
    while i > 0 or j > 0:
        b = back[i, j]
        if i > 0 and j > 0 and b == 0:
            if sim[i - 1][j - 1] >= thresh:
                pairs[i - 1] = (j - 1, sim[i - 1][j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and (b == 1 or j == 0):
            i -= 1
        else:
            j -= 1
    return pairs


def _merge_pass(flat, asr_words, pairs):
    """Handle split/merged words ('dil bar' sung as ASR 'dilbar'; ASR 'jaa na' for lyric 'jaana').

    Works only inside gaps between already-matched anchors. Returns {lyric_idx: (start, end, sim)}."""
    extra = {}
    anchors = sorted((i, j) for i, (j, _) in pairs.items())
    bounds = [(-1, -1)] + anchors + [(len(flat), len(asr_words))]
    for (i0, j0), (i1, j1) in zip(bounds, bounds[1:]):
        p, q = i0 + 1, j0 + 1
        while p < i1 and q < j1:
            done = False
            for n in (3, 2):                                   # several lyric words == one ASR word
                if p + n <= i1:
                    joined = "".join(flat[p:p + n])
                    sc = similarity(joined, asr_words[q].text)
                    if sc >= 0.8:
                        a = asr_words[q]
                        tot = float(sum(len(w) for w in flat[p:p + n]))
                        t = a.start
                        for k in range(n):
                            d = (a.end - a.start) * len(flat[p + k]) / tot
                            extra[p + k] = (t, t + d, sc); t += d
                        p += n; q += 1; done = True
                        break
            if done:
                continue
            if q + 1 < j1:                                      # one lyric word == two ASR words
                sc = similarity(flat[p], asr_words[q].text + asr_words[q + 1].text)
                if sc >= 0.8:
                    extra[p] = (asr_words[q].start, asr_words[q + 1].end, sc)
                    p += 1; q += 2
                    continue
            sc = similarity(flat[p], asr_words[q].text)
            if sc >= 0.55:
                extra[p] = (asr_words[q].start, asr_words[q].end, sc)
                p += 1; q += 1
            else:
                q += 1
    return extra



_ENGLISH_TRANSLATION_WORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "are",
    "as", "at", "away", "back", "be", "because", "been", "before", "being",
    "believe", "between", "both", "but", "by", "can", "come", "could", "day",
    "did", "do", "down", "each", "even", "every", "fall", "far", "feel", "find",
    "fire", "for", "from", "front", "get", "give", "go", "going", "gone", "good",
    "got", "had", "has", "have", "he", "heart", "her", "here", "him", "his",
    "hold", "hope", "how", "i", "if", "in", "into", "is", "it", "just", "know",
    "let", "like", "little", "live", "look", "love", "make", "man", "many", "me",
    "more", "most", "my", "name", "never", "night", "no", "not", "now", "of",
    "oh", "on", "once", "one", "only", "or", "other", "our", "out", "over",
    "people", "place", "right", "said", "same", "say", "see", "she", "should",
    "so", "someone", "something", "still", "story", "take", "tell", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "thing",
    "think", "this", "those", "through", "time", "to", "together", "too", "touch",
    "up", "us", "very", "was", "way", "we", "weather", "what", "when", "where",
    "which", "who", "why", "will", "with", "within", "without", "would", "you",
    "your", "happened", "recognizes", "talking", "ahead", "calm", "down", "dew",
    "hands", "hand", "front", "between", "someone", "knows", "beautiful",
}


def _looks_like_english_translation(text):
    """Conservative heuristic for plain-English translation lines.

    Romanized Hindi is intentionally not classified from one coincidental word
    such as 'are' or 'hai'. Require multiple English vocabulary hits and a
    meaningful fraction of the line to be English.
    """
    import re
    from .text_utils import has_devanagari

    if has_devanagari(text):
        return False
    tokens = [re.sub(r"[^a-z']", "", t.lower()) for t in text.split()]
    tokens = [t for t in tokens if t]
    if len(tokens) < 3:
        return False
    hits = sum(t in _ENGLISH_TRANSLATION_WORDS for t in tokens)
    return hits >= 2 and hits / len(tokens) >= 0.45


def _line_expected_duration(tokens):
    """Conservative fallback duration for a lyric line, never an unbounded gap."""
    weight = sum(max(1, syllables(t)) for t in tokens)
    return max(0.8, min(5.5, 0.34 * weight + 0.2))


def align_with_asr(lines: List[LyricLine], asr_words, audio_duration):
    """Align sung lines to ASR; English translation rows inherit Hindi row timing.

    User spelling is preserved. Missing words are estimated locally instead of
    being interpolated across the entire flattened lyric sheet.
    """
    from .text_utils import has_devanagari

    count = len(lines)
    translations = {}
    is_translation = [False] * count

    # Pair an obvious English translation only with the immediately preceding
    # non-translation line. Unpaired vocalizations and repeated chorus lines
    # remain part of the sung text.
    for li in range(1, count):
        if _looks_like_english_translation(lines[li].text):
            prev = li - 1
            if not is_translation[prev] and not _looks_like_english_translation(lines[prev].text):
                translations[li] = prev
                is_translation[li] = True

    flat, owner, line_token_indices = [], [], [[] for _ in lines]
    for li, ln in enumerate(lines):
        if is_translation[li]:
            continue
        for tok in _split_words(ln.text):
            line_token_indices[li].append(len(flat))
            flat.append(tok)
            owner.append(li)

    if not flat:
        raise RuntimeError("ASR output did not match the lyrics at all; check the language or the audio")

    # Discard only extremely low-confidence ASR words before matching.
    # Otherwise, a hallucinated repeated word far away in the audio can
    # become an anchor and stretch the lyric line across a large silence.
    reliable_asr_words = []
    weak_asr_words = []
    for word in asr_words or []:
        try:
            probability = float(getattr(word, "prob", 1.0))
        except (TypeError, ValueError, OverflowError):
            probability = 0.0

        if not np.isfinite(probability):
            continue

        if probability >= 0.35:
            reliable_asr_words.append(word)
        elif probability >= 0.0:
            # Retain weak words only as possible coarse timing boundaries.
            # They are never admitted as ordinary word-matching anchors.
            weak_asr_words.append(word)

    asr_words = reliable_asr_words
    if not asr_words:
        raise RuntimeError("ASR output did not match the lyrics at all; check the language or the audio")

    pairs = _nw_align(flat, asr_words)
    merged = _merge_pass(flat, asr_words, pairs)

    # Keep only chronological, non-overlapping ASR anchors. Merge-pass results
    # already contain timestamps; ordinary matches refer to ASR word indices.
    candidates = {}

    # Reject ASR anchors outside the known audio interval. Do not clamp an
    # invalid match onto the audio boundary: that could create a false match.
    duration = (
        float(audio_duration)
        if audio_duration is not None and float(audio_duration) > 0
        else None
    )

    def valid_anchor(start, end):
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError, OverflowError):
            return False

        import math

        if not (math.isfinite(start) and math.isfinite(end)):
            return False
        if start < 0 or end <= start:
            return False
        if duration is not None and (start >= duration or end > duration):
            return False
        return True

    def plausible_word_anchor(start, end, lyric_word):
        """Reject word matches whose duration is implausibly long.

        This is a conservative guard against ASR timestamp hallucinations.
        It does not shorten a suspicious timestamp and pretend it is correct;
        the word is instead interpolated and flagged for review.
        """
        span = float(end) - float(start)
        token = str(lyric_word).strip(".,!?;:()[]{}")

        # A single token may be sustained, but a multi-second timestamp for
        # a short token is more likely to be a bad ASR alignment.
        max_duration = min(3.5, max(1.5, 0.45 * max(1, len(token))))
        return span <= max_duration

    for i, (j, score) in pairs.items():
        aw = asr_words[j]
        if (
            valid_anchor(aw.start, aw.end)
            and plausible_word_anchor(aw.start, aw.end, flat[i])
        ):
            candidates[i] = (float(aw.start), float(aw.end), float(score))

    for i, (a, b, score) in merged.items():
        if (
            i not in candidates
            and valid_anchor(a, b)
            and plausible_word_anchor(a, b, flat[i])
        ):
            candidates[i] = (float(a), float(b), float(score))

    anchors = {}
    last_start = -1.0
    last_end = -1.0
    for i, (a, b, score) in sorted(candidates.items()):
        if a < last_start or a < last_end - 0.02:
            continue
        anchors[i] = (a, b, score)
        last_start, last_end = a, b

    if not anchors:
        raise RuntimeError(
            "ASR output did not match the lyrics at all; check the language or the audio"
        )

    token_times = {}
    for i, (a, b, score) in anchors.items():
        token_times[i] = (a, b, score, "asr")

    # Fill gaps only inside each line. Leading/trailing missing tokens receive
    # conservative estimates adjacent to that line's actual anchors.
    for li, indices in enumerate(line_token_indices):
        if not indices:
            continue
        line_anchors = [i for i in indices if i in token_times]
        if not line_anchors:
            continue
        first, last = line_anchors[0], line_anchors[-1]

        def fill_indices(missing, left, right):
            if not missing:
                return
            weights = [max(1, syllables(flat[k])) for k in missing]
            total = float(sum(weights))
            span = max(0.0, right - left)
            cursor = left
            for k, wt in zip(missing, weights):
                duration = span * wt / total if total else 0.1
                token_times[k] = (cursor, cursor + duration, 0.0, "interpolated")
                cursor += duration

        # Internal gaps: interpolate only between adjacent anchors on this line.
        for a_idx, b_idx in zip(line_anchors, line_anchors[1:]):
            missing = [k for k in indices if a_idx < k < b_idx and k not in token_times]
            if missing:
                fill_indices(missing, token_times[a_idx][1], token_times[b_idx][0])

        # Leading words: estimate backwards from the first matched word.
        missing_before = [k for k in indices if k < first and k not in token_times]
        if missing_before:
            dur = sum(max(1, syllables(flat[k])) for k in missing_before) * 0.32
            left = max(0.0, token_times[first][0] - min(dur, 4.0))
            fill_indices(missing_before, left, token_times[first][0])

        # Trailing words: estimate forwards from the last matched word.
        missing_after = [k for k in indices if k > last and k not in token_times]
        if missing_after:
            dur = sum(max(1, syllables(flat[k])) for k in missing_after) * 0.32
            right = token_times[last][1] + min(dur, 4.0)
            if audio_duration:
                right = min(float(audio_duration), right)
            fill_indices(missing_after, token_times[last][1], max(token_times[last][1] + 0.05, right))

    # Identify each non-translation line's real anchor interval.
    line_bounds = [None] * count
    for li, indices in enumerate(line_token_indices):
        if is_translation[li] or not indices:
            continue
        known = [token_times[i] for i in indices if i in token_times]
        if known:
            line_bounds[li] = (min(x[0] for x in known), max(x[1] for x in known))

    # Fill entire unmatched-line runs using bounded line durations, not a single
    # interpolation spanning arbitrary distances between ASR anchors.
    expected = {
        li: _line_expected_duration(_split_words(lines[li].text))
        for li in range(count) if not is_translation[li]
    }
    non_translation = [li for li in range(count) if not is_translation[li]]
    known_lines = [li for li in non_translation if line_bounds[li] is not None]

    for pos, li in enumerate(non_translation):
        if line_bounds[li] is not None:
            continue
        prevs = [x for x in known_lines if x < li]
        nexts = [x for x in known_lines if x > li]
        prev_line = prevs[-1] if prevs else None
        next_line = nexts[0] if nexts else None
        if prev_line is None and next_line is None:
            continue
        if prev_line is None:
            # Leading unmatched lines start at zero and share the available
            # lead-in if their expected durations do not fit.
            run = [x for x in non_translation if x < next_line and line_bounds[x] is None]
            total = sum(expected[x] for x in run)
            limit = max(0.0, line_bounds[next_line][0])
            # Keep leading lines visible even when the first matched anchor
            # occurs at or near zero. Never scale an entire run to zero.
            scale = min(1.0, limit / total) if total and limit >= 0.55 * len(run) else 1.0
            cursor = max(0.0, limit - total * scale)
            for x in run:
                dur = max(0.55, expected[x] * scale)
                line_bounds[x] = (cursor, cursor + dur)
                cursor += dur
        elif next_line is None:
            run = [x for x in non_translation if x > prev_line and line_bounds[x] is None]
            cursor = line_bounds[prev_line][1]
            for x in run:
                dur = expected[x]
                if audio_duration:
                    dur = min(dur, max(0.2, float(audio_duration) - cursor))
                line_bounds[x] = (cursor, cursor + dur)
                cursor += dur
        else:
            run = [x for x in non_translation if prev_line < x < next_line and line_bounds[x] is None]
            left, right = line_bounds[prev_line][1], line_bounds[next_line][0]
            total = sum(expected[x] for x in run)
            available = max(0.0, right - left)
            # Do not compress an unmatched line run to zero duration. If the
            # anchors leave too little room, preserve plausible line durations
            # and place the run immediately before the next anchor. This may
            # overlap a neighboring interval, but is reviewable and visible;
            # zero-width lines are not useful subtitles.
            minimum_useful = min(total, 0.55 * len(run))
            if available >= minimum_useful and total > 0:
                scale = min(1.0, available / total)
                cursor = left
            else:
                scale = 1.0
                cursor = max(0.0, right - total)
            for x in run:
                dur = max(0.55, expected[x] * scale)
                line_bounds[x] = (cursor, cursor + dur)
                cursor += dur

    # Any still-unresolved line (e.g. empty input) gets a safe short fallback.
    for li in non_translation:
        if line_bounds[li] is None:
            line_bounds[li] = (0.0, expected[li])

    # Materialize base events and reports in original lyric order.
    events_by_line, reports_by_line = {}, {}
    for li in non_translation:
        toks = _split_words(lines[li].text)
        indices = line_token_indices[li]
        if not toks or not indices:
            continue
        line_start, line_end = line_bounds[li]
        ws = []
        sources, confs = [], []
        for k, tok in zip(indices, toks):
            if k in token_times:
                a, b, score, source = token_times[k]
                sources.append(source)
                confs.append(score if source == "asr" else 0.35)
            else:
                a = b = None
                source, score = "interpolated", 0.0
                sources.append(source)
                confs.append(0.3)
            if a is None:
                ws.append(None)
            else:
                ws.append(Word(tok, float(a), float(b)))
        # Fill words missing from wholly unmatched lines across their bounded line interval.
        missing_pos = [j for j, w in enumerate(ws) if w is None]
        if missing_pos:
            estimated = _distribute([toks[j] for j in missing_pos], line_start, max(line_start + 0.2, line_end))
            for j, word in zip(missing_pos, estimated):
                ws[j] = word
        # Clamp and enforce monotonicity without creating zero-length words.
        for j, w in enumerate(ws):
            w.start = max(0.0, float(w.start))
            w.end = max(w.start + 0.05, float(w.end))
            if j and w.start < ws[j - 1].end:
                if w.start >= ws[j - 1].start:
                    ws[j - 1].end = max(ws[j - 1].start + 0.05, w.start)
                else:
                    w.start = ws[j - 1].end
                    w.end = max(w.start + 0.05, w.end)
        line_start = ws[0].start
        line_end = ws[-1].end
        anchored_count = sum(1 for k in indices if k in anchors)
        confidence = sum(anchors[k][2] for k in indices if k in anchors) / max(1, len(toks))
        notes = []
        if anchored_count < len(toks):
            notes.append(f"{len(toks) - anchored_count} of {len(toks)} words had no match in the audio transcript; timing estimated")
        reliable = anchored_count == len(toks)
        ev = LyricEvent(line_start, line_end, ws, None, reliable)
        events_by_line[li] = ev
        winfo, unc = _word_info(ws, sources, confs)
        reports_by_line[li] = LineReport(
            li, lines[li].text, line_start, line_end, "asr",
            round(confidence, 2), not reliable or confidence < 0.7,
            notes, winfo, unc
        )

    # Translation rows inherit their paired Hindi line's event window. Their
    # individual word positions are distributed over that same window and
    # marked estimated; the displayed English spelling remains untouched.
    for translation_li, source_li in translations.items():
        source_event = events_by_line.get(source_li)
        toks = _split_words(lines[translation_li].text)
        if source_event is None or not toks:
            continue
        ws = _distribute(toks, source_event.start, max(source_event.start + 0.2, source_event.end))
        ev = LyricEvent(source_event.start, source_event.end, ws, None, False)
        events_by_line[translation_li] = ev
        winfo, unc = _word_info(ws, ["interpolated"] * len(ws), [0.35] * len(ws))
        reports_by_line[translation_li] = LineReport(
            translation_li, lines[translation_li].text, ev.start, ev.end, "asr",
            0.35, True,
            [f"English translation paired with line {source_li + 1}; line timing inherited, word timing estimated"],
            winfo, unc
        )

    # Repair consecutive repeated lyric blocks when fallback timing causes
    # the block to overlap the preceding lyric. Reflow only the repeated
    # block, retain its credible end boundary, and flag all adjusted words.
    repeat_groups = []
    pos = 0

    while pos < len(non_translation):
        first_li = non_translation[pos]
        key = " ".join(lines[first_li].text.casefold().split())
        stop = pos + 1

        while stop < len(non_translation):
            candidate_li = non_translation[stop]
            candidate_key = " ".join(lines[candidate_li].text.casefold().split())
            if candidate_key != key:
                break
            stop += 1

        if stop - pos >= 2:
            repeat_groups.append((pos, non_translation[pos:stop]))

        pos = stop

    for group_pos, group in repeat_groups:
        if any(li not in events_by_line for li in group):
            continue

        previous_event = None
        if group_pos > 0:
            previous_li = non_translation[group_pos - 1]
            previous_event = events_by_line.get(previous_li)

        group_events = [events_by_line[li] for li in group]
        group_start = min(ev.start for ev in group_events)
        group_end = max(ev.end for ev in group_events)

        # Low-confidence words cannot anchor individual lyric words, but a
        # plausible matching token far after a repeated block may provide a
        # coarse end boundary. This prevents the block from being compressed
        # prematurely while avoiding a huge word-level interpolation gap.
        group_tokens = [
            token
            for repeated_li in group
            for token in _split_words(lines[repeated_li].text)
        ]
        soft_tail_ends = []

        for weak_word in weak_asr_words:
            try:
                weak_start = float(weak_word.start)
                weak_end = float(weak_word.end)
                weak_text = str(weak_word.text).strip(".,!?;:()[]{}")
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue

            if not (
                np.isfinite(weak_start)
                and np.isfinite(weak_end)
                and weak_start >= group_start
                and weak_end > weak_start
                and weak_end > group_end + 5.0
            ):
                continue

            if duration is not None and weak_end > duration:
                continue

            if any(similarity(token, weak_text) >= 0.55 for token in group_tokens):
                soft_tail_ends.append(weak_end)

        if soft_tail_ends:
            soft_end = max(soft_tail_ends)
            if duration is not None:
                soft_end = min(soft_end, duration)
            group_end = max(group_end, soft_end)

        # Only reflow a repeated block when its line intervals overlap
        # the preceding lyric or one another. Do not rewrite already ordered
        # word timings merely because some words were interpolated.
        has_internal_overlap = any(
            left.end > right.start + 0.02
            for left, right in zip(group_events, group_events[1:])
        )
        report_intervals = [
            (reports_by_line[li].start, reports_by_line[li].end)
            for li in group
        ]
        reports_are_ordered = all(
            left[1] <= right[0] + 0.02
            for left, right in zip(report_intervals, report_intervals[1:])
        )

        if reports_are_ordered and not has_internal_overlap:
            continue

        # The first repeated block has no preceding event. Preserve its
        # earliest detected start instead of skipping overlap repair entirely.
        new_start = (
            max(0.0, float(previous_event.end))
            if previous_event is not None
            else max(0.0, float(group_start))
        )
        new_end = max(float(group_end), new_start + 0.20 * len(group))
        if audio_duration is not None:
            new_end = min(float(audio_duration), new_end)

        if new_end <= new_start:
            continue

        line_span = (new_end - new_start) / len(group)

        for offset, li in enumerate(group):
            ev = events_by_line[li]
            start = new_start + offset * line_span
            end = new_start + (offset + 1) * line_span

            ev.words = _distribute(
                [word.text for word in ev.words],
                start,
                end,
            )
            ev.start = start
            ev.end = end
            ev.timing_reliable = False

            report = reports_by_line[li]
            report.start = start
            report.end = end
            report.needs_review = True
            report.confidence = min(report.confidence, 0.35)
            note = (
                "Repeated-line timing redistributed to remove overlap; "
                "review word timing."
            )
            if note not in report.notes:
                report.notes.append(note)

            report.words, report.uncertain_words = _word_info(
                ev.words,
                ["interpolated"] * len(ev.words),
                [0.35] * len(ev.words),
            )

            # Keep any paired English translation synchronized with its source.
            for translation_li, source_li in translations.items():
                if source_li != li or translation_li not in events_by_line:
                    continue

                translated = events_by_line[translation_li]
                translated_tokens = _split_words(lines[translation_li].text)
                translated.words = _distribute(translated_tokens, start, end)
                translated.start = start
                translated.end = end

                translated_report = reports_by_line[translation_li]
                translated_report.start = start
                translated_report.end = end
                translated_report.needs_review = True
                translated_report.notes.append(
                    "Source repeated-line timing was redistributed; review timing."
                )
                translated_report.words, translated_report.uncertain_words = _word_info(
                    translated.words,
                    ["interpolated"] * len(translated.words),
                    [0.35] * len(translated.words),
                )

    # Repair overlaps between consecutive base lyric events.
    # Keep each event's words and review-report timestamps inside its line.
    ordered_base = [
        li for li in non_translation if li in events_by_line
    ]

    previous_li = None
    for li in ordered_base:
        current = events_by_line[li]

        if previous_li is not None:
            previous = events_by_line[previous_li]

            if current.start < previous.end - 0.02:
                if current.start > previous.start + 0.05:
                    boundary = current.start
                    previous.end = boundary

                    # Clamp previous-line words to the shortened line.
                    for word in previous.words:
                        word.start = max(previous.start, min(word.start, boundary))
                        word.end = max(word.start + 0.01, min(word.end, boundary))
                        if word.end > boundary:
                            word.end = boundary
                            word.start = max(previous.start, boundary - 0.01)

                    previous_report = reports_by_line[previous_li]
                    previous_report.end = boundary
                    previous_report.needs_review = True
                    note = "Line end shortened to prevent overlap; review timing."
                    if note not in previous_report.notes:
                        previous_report.notes.append(note)
                else:
                    shift = previous.end - current.start
                    current.start += shift
                    current.end += shift
                    for word in current.words:
                        word.start += shift
                        word.end += shift

                    current_report = reports_by_line[li]
                    current_report.start = current.start
                    current_report.end = current.end
                    current_report.needs_review = True
                    note = "Line shifted to prevent overlap; review timing."
                    if note not in current_report.notes:
                        current_report.notes.append(note)

        # Keep word timestamps within the event after any repair.
        for word in current.words:
            word.start = max(current.start, min(word.start, current.end))
            word.end = max(word.start + 0.01, min(word.end, current.end))
            if word.end > current.end:
                word.start = max(current.start, current.end - 0.01)
                word.end = current.end

        report = reports_by_line[li]
        report.start = current.start
        report.end = current.end
        previous_li = li

    # Keep paired English translations synchronized with their source.
    for translation_li, source_li in translations.items():
        if translation_li not in events_by_line or source_li not in events_by_line:
            continue
        source_event = events_by_line[source_li]
        translated_event = events_by_line[translation_li]
        translated_event.start = source_event.start
        translated_event.end = source_event.end
        translated_report = reports_by_line[translation_li]
        translated_report.start = source_event.start
        translated_report.end = source_event.end

    events, reports = [], []
    for li in range(count):
        if li in events_by_line:
            events.append(events_by_line[li])
            reports.append(reports_by_line[li])
    _finalize_ends(events, audio_duration=audio_duration)

    # Synchronize report timestamps and clamp words to finalized event boundaries.
    for li, ev in enumerate(events):
        report = reports[li]
        report.start = ev.start
        report.end = ev.end

        def clamp_word_times(word, fallback_start, fallback_end):
            start = word.get("start")
            end = word.get("end")

            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                start, end = fallback_start, fallback_end

            start = max(ev.start, min(float(start), ev.end))
            end = max(start, min(float(end), ev.end))

            word["start"] = round(start, 3)
            word["end"] = round(end, 3)

        for i, word in enumerate(report.words):
            if i < len(ev.words):
                word["start"] = ev.words[i].start
                word["end"] = ev.words[i].end
            clamp_word_times(word, ev.start, ev.end)

        for word in report.uncertain_words:
            word_index = word.get("i")
            if isinstance(word_index, int) and 0 <= word_index < len(ev.words):
                word["start"] = ev.words[word_index].start
                word["end"] = ev.words[word_index].end
            clamp_word_times(word, ev.start, ev.end)

    # _finalize_ends normally gives each subtitle a hold interval. Paired
    # translations must exactly share their source line's visibility window.
    for translation_li, source_li in translations.items():
        if translation_li in events_by_line and source_li in events_by_line:
            source_event = events_by_line[source_li]
            translated_event = events_by_line[translation_li]
            translated_event.start = source_event.start
            translated_event.end = source_event.end
            translated_report = reports_by_line[translation_li]
            translated_report.start = source_event.start
            translated_report.end = source_event.end

    # Clamp report word timestamps AFTER translation visibility windows are finalized.
    for ev, report in zip(events, reports):
        report.start = ev.start
        report.end = ev.end

        for word in report.words:
            start = word.get("start", ev.start)
            end = word.get("end", ev.end)
            start = max(ev.start, min(float(start), ev.end))
            end = max(start, min(float(end), ev.end))
            word["start"] = round(start, 3)
            word["end"] = round(end, 3)

        for word in report.uncertain_words:
            start = word.get("start", ev.start)
            end = word.get("end", ev.end)
            start = max(ev.start, min(float(start), ev.end))
            end = max(start, min(float(end), ev.end))
            word["start"] = round(start, 3)
            word["end"] = round(end, 3)

    return events, reports


# ------------------------------------------------------------------ 3. energy estimate
def voiced_regions(samples, hop=0.02, min_gap=0.30):
    """Rough 'something is happening' regions from short-time energy. Returns (regions, env, hop)."""
    h = int(SR * hop)
    n = len(samples) // h
    if n < 10:
        return [], np.zeros(0), hop
    env = np.sqrt((samples[:n * h].reshape(n, h) ** 2).mean(axis=1))
    k = max(1, int(0.12 / hop))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    thr = max(1e-4, 0.25 * np.percentile(env, 90))
    on = env > thr
    regions, s = [], None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        elif not v and s is not None:
            regions.append([s * hop, i * hop]); s = None
    if s is not None:
        regions.append([s * hop, n * hop])
    merged = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        elif r[1] - r[0] > 0.12:
            merged.append(r)
    return merged, env, hop


def align_estimated(lines: List[LyricLine], samples, audio_duration, window=None):
    """Map syllables onto voiced time. Roughly right for sparse/vocal-forward audio, poor with
    dense instrumentals: every line is flagged 'estimated' and needs review."""
    regions, _, _ = voiced_regions(samples) if samples is not None else ([], None, None)
    if window:                                   # user-supplied 'vocals are between these times'
        w0, w1 = window
        regions = [[max(a, w0), min(b, w1)] for a, b in regions if b > w0 and a < w1]
    if not regions:
        regions = [[0.0, audio_duration or 30.0]]
    voiced_len = sum(b - a for a, b in regions)
    sylls = [[max(1, syllables(t)) for t in _split_words(l.text)] for l in lines]
    total = float(sum(sum(s) for s in sylls))

    def to_time(v):                              # v in voiced seconds -> real seconds
        acc = 0.0
        for a, b in regions:
            if v <= acc + (b - a):
                return a + (v - acc)
            acc += b - a
        return regions[-1][1]

    events, reports, cum = [], [], 0.0
    for li, (ln, sy) in enumerate(zip(lines, sylls)):
        toks = _split_words(ln.text)
        ws = []
        for tok, s in zip(toks, sy):
            a = to_time(cum / total * voiced_len); cum += s
            b = to_time(cum / total * voiced_len)
            ws.append(Word(tok, a, max(b, a + 0.05)))
        events.append(LyricEvent(ws[0].start, ws[-1].end, ws, None, False))
        winfo, unc = _word_info(ws, ["estimated"] * len(ws), [0.3] * len(ws))
        reports.append(LineReport(li, ln.text, ws[0].start, ws[-1].end, "estimated", 0.3, True,
                                  ["timing estimated from audio energy only; add LRC timestamps or install faster-whisper"],
                                  winfo, unc))
    _finalize_ends(events, audio_duration=audio_duration)
    return events, reports


def _finalize_ends(events, hold=1.2, audio_duration=None):
    """Finalize subtitle visibility without exceeding the audio duration."""
    duration = None
    if audio_duration is not None:
        try:
            candidate = float(audio_duration)
            if np.isfinite(candidate) and candidate > 0:
                duration = candidate
        except (TypeError, ValueError, OverflowError):
            duration = None

    for i, ev in enumerate(events):
        if not ev.words:
            continue

        last_end = max(float(w.end) for w in ev.words)
        nxt = events[i + 1].start if i + 1 < len(events) else None
        ev.start = max(0.0, float(ev.words[0].start))

        candidate_end = min(nxt, last_end + hold) if nxt is not None else last_end + hold
        ev.end = max(candidate_end, last_end, ev.start + 0.55)

        if duration is not None:
            ev.start = min(ev.start, duration)
            ev.end = min(ev.end, duration)

        # Keep every word inside its event's final visibility interval.
        for word in ev.words:
            word.start = max(ev.start, float(word.start))
            word.end = min(ev.end, float(word.end))
            if word.end <= word.start:
                word.end = min(ev.end, word.start + 0.05)


def align_lyrics(lines, timed, audio=None, asr_words=None, audio_duration=None, window=None):
    """Dispatch to the best available timing source. Returns (events, reports, method)."""
    if timed:
        # Synced lyric catalogs usually provide line starts, not karaoke word
        # timestamps. Use the LRC text as the spelling source and ASR words as
        # vocal timing anchors whenever possible.
        if asr_words:
            ev, rp = align_with_asr(lines, asr_words, audio_duration)
            for line, event, report in zip(lines, ev, rp):
                report.method = "lrc+asr"
                if line.start is None:
                    continue
                delta = float(line.start) - float(event.start)
                if abs(delta) <= 0.75:
                    for word in event.words:
                        word.start = max(0.0, word.start + delta)
                        word.end = max(word.start + 0.05, word.end + delta)
                    event.start = event.words[0].start
                    event.end = event.words[-1].end
                    report.start, report.end = event.start, event.end
                    for word in report.words:
                        word["start"] = round(max(0.0, word["start"] + delta), 3)
                        word["end"] = round(max(word["start"] + 0.05, word["end"] + delta), 3)
                else:
                    report.needs_review = True
                    note = "LRC line start differs from the recognized vocal timing by more than 0.75 seconds."
                    if note not in report.notes:
                        report.notes.append(note)
            return ev, rp, "lrc+asr"
        ev, rp = align_from_lrc(lines, audio_duration)
        return ev, rp, "lrc"
    if asr_words:
        ev, rp = align_with_asr(lines, asr_words, audio_duration)
        return ev, rp, "asr"
    samples = load_audio(audio) if audio else None
    ev, rp = align_estimated(lines, samples, audio_duration, window)
    return ev, rp, "estimated"
