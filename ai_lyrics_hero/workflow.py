"""High-level steps: lyrics(+audio) -> ASS + review report; and the one-shot 'make' video."""
import json
import logging
import math
import os
import re
import copy
from dataclasses import asdict
from . import asr as asr_mod
from .align import align_lyrics, LineReport
from .ass_writer import write_ass
from .audio import load_audio, SR
from .config import HeroConfig
from .hero import select_heroes
from .imagery import Lexicon, plan_cues
from .scenes import SceneLexicon, plan_scenes
from .lyrics import parse_lyrics, LyricLine
from .lrclib import lookup_lyrics as lookup_track_lyrics
from .pipeline import probe, render_video
from .text_utils import romanize_hindi

log = logging.getLogger("ai_lyrics_hero")


def _draft_lines_from_asr(words, gap=0.7, max_words=8):
    lines, cur = [], []
    for w in words:
        if cur and (w.start - cur[-1].end > gap or len(cur) >= max_words):
            lines.append(cur); cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return [LyricLine(romanize_hindi(" ".join(x.text for x in l))) for l in lines]


def _write_auto_lrc(reports, ass_path):
    """Save generated lyrics and their word timing as editable LRC beside the ASS."""
    path = os.path.splitext(os.fspath(ass_path))[0] + ".auto.lrc"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def stamp(seconds):
        total = max(0, int(round(float(seconds) * 100)))
        minutes, remainder = divmod(total, 6000)
        whole_seconds, centiseconds = divmod(remainder, 100)
        return f"{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"

    with open(path, "w", encoding="utf-8") as f:
        for report in reports:
            words = report.words
            if words and len(words) == len(report.text.split()):
                body = " ".join(f"<{stamp(word['start'])},{stamp(word['end'])}>{word['text']}" for word in words)
            else:
                body = report.text
            f.write(f"[{stamp(report.start)}]{body}\n")
    return os.path.abspath(path)


def make_ass(audio, out_ass, lyrics_text=None, language=None, window=None, use_asr=True,
             model_size="medium", cfg=None, romanize=False, track_title=None, artist=None,
             lyrics_lookup=True):
    """Create the timed ASS file and a review report. Returns the report dict.

    Supplied lyrics are used verbatim. Automatic mode first tries a matching lyric catalog entry;
    if none is found, speech recognition creates a DRAFT whose spelling cannot be guaranteed."""
    cfg = cfg or HeroConfig()
    if audio:
        output_path = os.path.normcase(os.path.abspath(os.fspath(out_ass)))
        audio_path = os.path.normcase(os.path.abspath(os.fspath(audio)))
        if output_path == audio_path:
            raise ValueError(f"ASS output must not replace the audio input: {audio}")
    warnings = []
    duration = probe(audio)[0] if audio else None
    asr_words, draft = None, False
    lyrics_source = {"provider": "user", "kind": "provided"} if lyrics_text else None
    if not lyrics_text and lyrics_lookup and audio:
        found_lyrics, source_info, lookup_warning = lookup_track_lyrics(
            audio, duration, track_title, artist
        )
        if found_lyrics:
            lyrics_text = found_lyrics
            lyrics_source = source_info
            warnings.append(
                "Lyrics were matched from LRCLIB, a community-supplied catalog entry. Review spelling against the recording."
            )
        elif lookup_warning:
            warnings.append(lookup_warning)
    lines, timed = (parse_lyrics(lyrics_text) if lyrics_text else ([], False))
    if romanize:
        for line in lines:
            line.text = romanize_hindi(line.text)
    asr_error = None
    needs_asr_timing = bool(lines and timed and any(
        not line.word_starts or len(line.word_starts) != len(line.text.split()) for line in lines
    ))
    if use_asr and audio and (not lines or needs_asr_timing or not timed):
        try:
            if asr_mod.available():
                asr_words = asr_mod.transcribe(audio, language, model_size)
            else:
                asr_error = "faster-whisper is not installed; run `python -m pip install -r requirements.txt`"
        except Exception as e:                           # model download / decode problems
            asr_error = str(e)
            fallback = "; fell back to an energy-based estimate" if lines else "; no lyric transcript was produced"
            warnings.append(f"speech recognition failed ({e}){fallback}")
    if asr_error and use_asr:
        warnings.append("automatic lyrics need a working Faster-Whisper install and model cache: " + asr_error)
    if not lines:
        if not asr_words:
            reason = f" Automatic transcription error: {asr_error}." if asr_error else ""
            raise RuntimeError("Could not create lyrics automatically." + reason +
                               " Install the project requirements and try again, or supply --lyrics.")
        lines = _draft_lines_from_asr(asr_words)
        draft = True
        warnings.append("DRAFT transcript in Romanized Latin script from speech recognition: spelling and transliteration "
                        "are NOT guaranteed. "
                        "Correct the text, then re-run with --lyrics for exact spelling.")
        lyrics_source = {"provider": "speech_recognition", "kind": "unverified_draft"}
    elif lyrics_source is None:
        lyrics_source = {"provider": "speech_recognition", "kind": "unverified_draft"}
    events, reports, method = align_lyrics(lines, timed, audio, asr_words, duration, window)
    if method.startswith("lrc") and any(r.needs_review for r in reports):
        warnings.append("Some synced lyric lines have inferred or unmatched word timing; review the flagged lines in the report.")
    if method == "estimated":
        warnings.append("Timing is estimated from audio energy (typically +/- 0.5 s, worse with dense music). "
                        "Use --vocal-start/--vocal-end, LRC timestamps or faster-whisper for better timing.")
    if draft:
        for r in reports:
            r.needs_review = True
            r.notes.append("draft transcript - check spelling")
    else:
        auto_lyrics_path = None
    if draft or (lyrics_source and lyrics_source.get("provider") == "LRCLIB"):
        auto_lyrics_path = _write_auto_lrc(reports, out_ass)
        if draft:
            warnings.append(f"Auto-created word-timed draft lyrics: {auto_lyrics_path}. Review spelling before publishing.")
    select_heroes(events, cfg)
    cues, notes = plan_cues(events, Lexicon(cfg.lexicon_path or None), cfg) if cfg.images else ({}, [])
    scene_list, scene_notes = [], []
    if cfg.background == "scenes":
        plans, scene_notes = plan_scenes(events, SceneLexicon(cfg.scene_lexicon_path or None), cfg)
        scene_list = [{"line": i, "scene": p.kind, "word": p.word} for i, p in enumerate(plans)]
    write_ass(events, out_ass)
    report = {
        "scenes": scene_list, "scene_notes": scene_notes,
        "method": method, "audio_duration": duration, "warnings": warnings,
        "lyrics_source": lyrics_source,
        "auto_lyrics": auto_lyrics_path,
        "lines_needing_review": [r.index for r in reports if r.needs_review],
        "lines": [asdict(r) for r in reports],
        "images": [{"line": i, "word": c.word, "image": c.entry_id, "preset": c.preset, "time": round(c.time, 2)}
                   for i, c in sorted(cues.items())],
        "image_notes": notes,
    }
    write_review(report, out_ass)
    return report


def write_review(report, ass_path):
    base = os.path.splitext(ass_path)[0]
    with open(base + ".review.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    with open(base + ".review.txt", "w", encoding="utf-8") as f:
        f.write(format_review(report))


def format_review(rep):
    out = [f"Timing method: {rep['method']}", ""]
    source = rep.get("lyrics_source") or {}
    if source.get("provider") == "LRCLIB":
        out.append(f"Lyrics source: LRCLIB community catalog ({source.get('kind', 'lyrics')}); "
                   f"{source.get('track_name', '')} — {source.get('artist_name', '')}")
        out.append("")
    elif source.get("provider") == "speech_recognition":
        out.extend(["Lyrics source: speech recognition draft (spelling is unverified)", ""])
    if rep.get("auto_lyrics"):
        source = rep.get("lyrics_source") or {}
        label = "timed lyrics from the community catalog" if source.get("provider") == "LRCLIB" else "editable word-timed draft"
        out.append(f"Generated lyrics file: {rep['auto_lyrics']} ({label})")
        out.append("")
    r = rep.get("render")
    if r:
        m = r["measured"]
        out.append(f"Render: {r['final_seconds']}s = {r['main_seconds']}s lyrics + {r['outro_seconds']}s outro, "
                   f"{r['frames']} frames, encoder {r['encoder']}")
        if m:
            out.append(f"Verified: video {m.get('video_duration', 0):.3f}s, audio {m.get('audio_duration', 0):.3f}s, "
                       f"{m.get('width')}x{m.get('height')} @ {m.get('fps', 0):.2f} fps")
        out.append("")
    source_window = rep.get("source_window")
    if source_window:
        source_start = float(source_window["start"])
        source_end = float(source_window["end"])
        out.append(f"Song section used: [{_song_time(source_start)}-{_song_time(source_end)}] "
                   f"({source_start:.2f}-{source_end:.2f}s from the original song)")
        out.append("Lyrics in this section (original-song timestamps):")
        selected = [ln for ln in rep.get("lines", [])
                    if float(ln["end"]) > source_start and float(ln["start"]) < source_end]
        for ln in selected:
            line_start, line_end = float(ln["start"]), float(ln["end"])
            shown_start, shown_end = max(line_start, source_start), min(line_end, source_end)
            clipped = ""
            if shown_start > line_start or shown_end < line_end:
                clipped = f" (visible {_song_time(shown_start)}-{_song_time(shown_end)})"
            out.append(f"  {ln['text']} [{_song_time(line_start)}-{_song_time(line_end)}]{clipped}")
        if not selected:
            out.append("  No lyric lines overlap this section.")
        out.append("")
    selection = rep.get("auto_selection")
    if selection:
        out.append(f"Best lyric section selected from the full {rep.get('audio_duration', 0):.2f}s song: "
                   f"scanned {selection['candidates_scanned']} windows; lyric coverage "
                   f"{selection['lyric_coverage']:.0%}, {selection['word_density']:.2f} words/s, "
                   f"{selection['complete_lines']} complete lines, "
                   f"{selection['repeated_phrase_hits']} repeated phrases.")
        out.append("")
    for w in rep["warnings"]:
        out.append("WARNING: " + w)
    out.append("")
    for ln in rep["lines"]:
        flag = "  <-- REVIEW" if ln["needs_review"] else ""
        out.append(f"[{ln['start']:7.2f}-{ln['end']:7.2f}s | "
                   f"{_song_time(ln['start'])}-{_song_time(ln['end'])}] "
                   f"conf {ln['confidence']:.2f}  {ln['text']}{flag}")
        for n in ln["notes"]:
            out.append("      note: " + n)
        for u in ln.get("uncertain_words", []):
            out.append(f"      uncertain: '{u['text']}' ~{u['start']:.2f}s ({u['source']}, conf {u['confidence']})")
    if rep["images"]:
        out.append("\nImages that will appear:")
        for im in rep["images"]:
            out.append(f"  line {im['line']}: '{im['word']}' -> {im['image']} ({im['preset']}) at {im['time']}s")
    if rep.get("music"):
        mu = rep["music"]
        out.append(f"Music analysis: {'ok' if mu['reliable'] else 'fallback'}, tempo {mu['tempo'] or 'n/a'} BPM, "
                   f"{mu['beats']} beats, {mu['impacts']} impacts")
    if rep.get("visual_plan"):
        out.append(f"\nVisual plan (semantic analysis: {rep.get('semantic_source', 'local')}):")
        for v in rep["visual_plan"]:
            tr = f", in via {v['transition']}" if v["transition"] != "none" else ""
            out.append(f"  line {v['line']}: {v['scene']} ({v['source']}, {v['level']}), text={v['typography']}, "
                       f"layout={v['composition']}{tr}")
        for it in rep.get("interludes", []):
            out.append(f"  instrumental {it['start']}-{it['end']}s: {it['scene']} scene (no lyrics)")
    if rep.get("scenes"):
        out.append("\nBackground scene per line:")
        for sc in rep["scenes"]:
            why = f" (word '{sc['word']}')" if sc["word"] else ""
            out.append(f"  line {sc['line']}: {sc['scene']}{why}")
    for n in rep.get("scene_notes", []):
        out.append("  note: " + n)
    for n in rep["image_notes"]:
        out.append("  note: " + n)
    return "\n".join(out) + "\n"


def _song_time(seconds):
    """Format a timestamp as minutes:seconds.centiseconds in the original song."""
    total = max(0, int(round(float(seconds) * 100)))
    minutes, remainder = divmod(total, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def select_best_lyric_window(lines, audio_duration, window_duration, step=0.5):
    """Choose a continuous, lyric-dense window by scanning the entire song transcript."""
    audio_duration = float(audio_duration)
    window_duration = float(window_duration)
    step = max(0.1, float(step))
    if audio_duration <= 0 or window_duration <= 0:
        raise ValueError("audio and selected video duration must be positive")
    if window_duration > audio_duration + 1e-6:
        raise ValueError(f"video needs {window_duration:.2f}s of song, but the song is only {audio_duration:.2f}s")

    def normalize(value):
        return re.sub(r"\s+", " ", str(value or "").casefold()).strip()

    phrase_counts = {}
    for line in lines:
        phrase = normalize(line.get("text"))
        if phrase:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    usable = []
    for line in lines:
        try:
            start, end = float(line["start"]), float(line["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        words = []
        for word in line.get("words") or []:
            try:
                word_start, word_end = float(word["start"]), float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(word_start) and math.isfinite(word_end) and word_end > word_start:
                words.append((word_start, word_end, normalize(word.get("text"))))
        if not words:
            parts = str(line.get("text", "")).split()
            span = (end - start) / max(1, len(parts))
            words = [(start + i * span, start + (i + 1) * span, normalize(part))
                     for i, part in enumerate(parts)]
        try:
            confidence = max(0.0, min(1.0, float(line.get("confidence", 0.5) or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        usable.append({"start": start, "end": end, "phrase": normalize(line.get("text")),
                       "confidence": confidence, "words": words})
    if not usable:
        raise ValueError("No timed lyrics were found, so a best section cannot be selected.")

    def phrase_sequences(timed_words, max_gap=1.25):
        sequences, current, previous_end = [], [], None
        for word_start, word_end, word_text in sorted(timed_words):
            tokens = re.findall(r"\w+", word_text)
            if not tokens:
                continue
            if current and word_start - previous_end > max_gap:
                sequences.append(current)
                current = []
            current.extend(tokens)
            previous_end = word_end
        if current:
            sequences.append(current)
        return sequences

    all_words = [word for line in usable for word in line["words"]]
    repeated_phrases = {}
    for tokens in phrase_sequences(all_words):
        for size in (3, 4):
            for index in range(max(0, len(tokens) - size + 1)):
                phrase = tuple(tokens[index:index + size])
                repeated_phrases[phrase] = repeated_phrases.get(phrase, 0) + 1

    latest_start = max(0.0, audio_duration - window_duration)
    candidates = {round(i * step, 3) for i in range(int(latest_start / step) + 1)}
    candidates.add(round(latest_start, 3))
    for line in usable:
        for boundary in (line["start"], line["end"]):
            for offset in (-window_duration + 0.5, -window_duration / 2, -1.0, 0.0):
                candidates.add(round(max(0.0, min(latest_start, boundary + offset)), 3))

    best = None
    for start in sorted(candidates):
        end = start + window_duration
        intervals, words_in_window, overlapping = [], 0, []
        complete, exact_repeat_hits, clipping = 0, 0, 0.0
        window_words = []
        weighted_confidence = confidence_weight = 0.0
        for line in usable:
            overlap = max(0.0, min(end, line["end"]) - max(start, line["start"]))
            if overlap <= 0:
                continue
            overlapping.append(line)
            span = max(0.1, line["end"] - line["start"])
            if line["start"] >= start - 0.05 and line["end"] <= end + 0.05:
                complete += 1
            else:
                clipping += min(1.0, max(0.0, (span - overlap) / span))
            if line["phrase"] and phrase_counts.get(line["phrase"], 0) > 1:
                exact_repeat_hits += 1
            weighted_confidence += line["confidence"] * overlap
            confidence_weight += overlap
            for word_start, word_end, word_text in line["words"]:
                word_overlap = max(0.0, min(end, word_end) - max(start, word_start))
                if word_overlap > 0:
                    intervals.append((max(start, word_start), min(end, word_end)))
                    if start <= (word_start + word_end) / 2 < end:
                        words_in_window += 1
                        window_words.append((word_start, word_end, word_text))

        window_phrases = {}
        for tokens in phrase_sequences(window_words):
            for size in (3, 4):
                for index in range(max(0, len(tokens) - size + 1)):
                    phrase = tuple(tokens[index:index + size])
                    window_phrases[phrase] = window_phrases.get(phrase, 0) + 1

        intervals.sort()
        covered = 0.0
        if intervals:
            left, right = intervals[0]
            for next_left, next_right in intervals[1:]:
                if next_left <= right:
                    right = max(right, next_right)
                else:
                    covered += right - left
                    left, right = next_left, next_right
            covered += right - left
        coverage = min(1.0, covered / window_duration)
        density = words_in_window / window_duration
        confidence = weighted_confidence / confidence_weight if confidence_weight else 0.0
        complete_ratio = complete / max(1, len(overlapping))
        ngram_repeat_hits = sum(count for phrase, count in window_phrases.items()
                                if repeated_phrases.get(phrase, 0) > 1)
        repeat_hits = exact_repeat_hits + ngram_repeat_hits
        repeat_bonus = min(1.0, repeat_hits / 8.0)
        score = (0.42 * coverage + 0.30 * min(1.0, density / 1.5) + 0.10 * confidence +
                 0.03 * complete_ratio + 0.15 * repeat_bonus - min(0.15, clipping * 0.025))
        result = {"start": round(start, 3), "end": round(end, 3), "score": round(score, 5),
                  "lyric_coverage": round(coverage, 4), "word_density": round(density, 3),
                  "repeated_phrase_hits": repeat_hits, "complete_lines": complete,
                  "candidates_scanned": len(candidates)}
        if best is None or (result["score"], result["repeated_phrase_hits"], result["lyric_coverage"], -start) > \
                (best["score"], best["repeated_phrase_hits"], best["lyric_coverage"], -best["start"]):
            best = result
    if best is None or best["word_density"] <= 0:
        raise ValueError("No timed words were found across the song, so a best section cannot be selected.")
    return best


def make_video(audio, out_mp4, lyrics_text=None, cfg=None, outro=None, size=(1080, 1920), fps=30,
               start=0.0, end=None, encoder="auto", language=None, window=None, use_asr=True,
               model_size="medium", keep_ass=None, progress=None, romanize=False, auto_select=False,
               track_title=None, artist=None, lyrics_lookup=True):
    cfg = cfg or HeroConfig()
    ass_path = keep_ass or (os.path.splitext(out_mp4)[0] + ".ass")
    report = make_ass(audio, ass_path, lyrics_text, language, window, use_asr, model_size, cfg,
                      romanize=romanize, track_title=track_title, artist=artist,
                      lyrics_lookup=lyrics_lookup)
    render_cfg = cfg
    if auto_select:
        if cfg.target_duration:
            outro_seconds = 0.0
            if outro:
                outro_seconds = probe(outro)[0]
                if cfg.outro_duration:
                    outro_seconds = min(outro_seconds, cfg.outro_duration)
            desired_duration = cfg.target_duration - outro_seconds
        elif end is not None:
            desired_duration = float(end) - float(start)
        else:
            desired_duration = float(report["audio_duration"])
        selection = select_best_lyric_window(report["lines"], report["audio_duration"], desired_duration)
        start, end = selection["start"], selection["end"]
        render_cfg = copy.copy(cfg)
        render_cfg.target_duration = 0.0
        report["auto_selection"] = selection
    res = render_video(ass_path, audio, out_mp4, render_cfg, size, fps, start, end, outro, encoder, progress=progress)
    report["render"] = {"output": res.path, "encoder": res.encoder, "main_seconds": round(res.plan.main, 3),
                        "outro_seconds": round(res.plan.outro, 3), "final_seconds": round(res.plan.final, 3),
                        "frames": res.plan.total_frames, "measured": res.measured, "warnings": res.warnings}
    report["source_window"] = {
        "start": round(float(start), 3),
        "end": round(float(start + res.plan.main), 3),
    }
    report["warnings"] = report["warnings"] + [w for w in res.warnings if w not in report["warnings"]]
    d, m = res.direction, res.music
    if d is not None:
        report["visual_plan"] = [dict(line=p.index, scene=p.scene, source=p.source, mood=p.mood, level=p.level,
                                      typography=p.typography, composition=p.composition,
                                      transition=p.transition_in, user_asset=bool(p.asset_path)) for p in d.plans]
        report["interludes"] = [dict(start=round(i.start, 2), end=round(i.end, 2), scene=i.scene) for i in d.interludes]
        report["semantic_source"] = d.semantic_source
    if m is not None:
        report["music"] = dict(reliable=m.reliable, tempo=None if m.tempo is None else round(m.tempo, 1),
                               beats=len(m.beats), impacts=len(m.impacts), notes=m.notes)
    write_review(report, ass_path)                      # refresh the review files with the render results
    return report
