import argparse
import json
import logging
import sys
from .config import HeroConfig, STYLE_PRESETS
from .pipeline import render_video


def _common_render_args(ap):
    ap.add_argument("--config", help="JSON file with HeroConfig keys (all optional)")
    ap.add_argument("--title")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1080x1920")
    ap.add_argument("--outro")
    ap.add_argument("--target-duration", type=float, help="final length in seconds INCLUDING the outro (main = target - outro)")
    ap.add_argument("--outro-duration", type=float, help="use only this many seconds of the outro (default: all of it)")
    ap.add_argument("--outro-audio", choices=["continue", "outro", "silent", "mix"],
                    help="audio under the outro: continue the song (default) | outro's own audio | silence | crossfade mix")
    ap.add_argument("--crossfade", type=float, help="seconds of crossfade for --outro-audio mix")
    ap.add_argument("--seed", type=int, help="fixed seed for reproducible visuals (default: choose a stable look from the song)")
    ap.add_argument("--asset-dir", help="folder with your own backgrounds (see assets/user/README.txt); separate several with ;")
    ap.add_argument("--beat-intensity", type=float, help="0 = no beat pulses/shake, 1 = default, 2 = strong")
    ap.add_argument("--llm", choices=["none", "openai"], help="optional AI scene analysis (needs AI_LYRICS_LLM_API_KEY)")
    ap.add_argument("--composition", help="auto | centered | offset | bg_word | minimal")
    ap.add_argument("--typography", help="auto | pop | slide | rotate | wipe | tracking | blur | chars | impact")
    ap.add_argument("--transitions", help="auto | fade | zoom | slide | wipe | cut")
    ap.add_argument("--no-interludes", action="store_true", help="plain sky during instrumental breaks")
    ap.add_argument("--no-verify", action="store_true", help="skip the ffprobe/decode check of the finished file")
    ap.add_argument("--encoder", default="auto", choices=["auto", "libx264", "h264_qsv"])
    ap.add_argument("--style", choices=sorted(STYLE_PRESETS), help="look preset: classic | hero | scene (night scenes + captions) | scene-hero")
    ap.add_argument("--no-hero", action="store_true", help="use the plain lyric presentation")
    ap.add_argument("--no-images", action="store_true", help="disable meaning-matched illustrations")
    ap.add_argument("--allow-ambiguous", action="store_true", help="show images for multi-meaning words")


def _align_args(ap):
    ap.add_argument("--lyrics", help="optional text/LRC file; otherwise look up lyrics, then fall back to transcription")
    ap.add_argument("--romanize", action="store_true",
                    help="display Devanagari Hindi in Latin letters (transliteration, not translation)")
    ap.add_argument("--language", help="language hint for speech recognition, e.g. hi, en")
    ap.add_argument("--vocal-start", type=float, help="seconds where the singing starts (helps the estimate)")
    ap.add_argument("--vocal-end", type=float, help="seconds where the singing ends")
    ap.add_argument("--no-asr", action="store_true", help="do not use speech recognition")
    ap.add_argument("--model", default="medium", help="faster-whisper model size (medium is more accurate, but slower)")
    ap.add_argument("--track-title", help="song title for the exact lyric catalog lookup")
    ap.add_argument("--artist", help="song artist for the exact lyric catalog lookup")
    ap.add_argument("--no-lyrics-lookup", action="store_true", help="skip the LRCLIB metadata lookup")


def _cfg(a):
    data = json.load(open(a.config, encoding="utf-8")) if a.config else {}
    if getattr(a, "style", None):
        data.update(STYLE_PRESETS[a.style])
    if a.no_hero:
        data["enabled"] = False
    if a.no_images:
        data["images"] = False
    if a.allow_ambiguous:
        data["allow_ambiguous"] = True
    if a.title:
        data["title"] = a.title
    for flag, key in (("target_duration", "target_duration"), ("outro_duration", "outro_duration"),
                      ("outro_audio", "outro_audio"), ("crossfade", "crossfade_s"), ("seed", "seed")):
        v = getattr(a, flag, None)
        if v is not None:
            data[key] = v
    for flag, key in (("asset_dir", "asset_dirs"), ("beat_intensity", "beat_intensity"), ("llm", "llm_provider"),
                      ("composition", "composition"), ("typography", "typography"), ("transitions", "transitions")):
        v = getattr(a, flag, None)
        if v is not None:
            data[key] = v
    if getattr(a, "no_interludes", False):
        data["interludes"] = False
    if getattr(a, "no_verify", False):
        data["verify_output"] = False
    return HeroConfig.from_dict(data)


def _window(a):
    if getattr(a, "vocal_start", None) is None and getattr(a, "vocal_end", None) is None:
        return None
    start = 0.0 if a.vocal_start is None else a.vocal_start
    end = a.vocal_end if a.vocal_end is not None else 1e9
    if start < 0:
        raise ValueError("--vocal-start must be >= 0")
    if end <= start:
        raise ValueError("--vocal-end must be later than --vocal-start")
    return start, end


def _read(path):
    return open(path, encoding="utf-8-sig").read() if path else None


def _parse_size(value):
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("--size must be written as WIDTHxHEIGHT, for example 1080x1920") from None
    if width < 2 or height < 2:
        raise ValueError("--size width and height must both be at least 2 pixels")
    return width, height


def main(argv=None):
    # ASR drafts can contain replacement or non-Latin characters that the Windows legacy
    # console code page cannot print; reports are still written losslessly as UTF-8 files.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("align", "make"):
        return _main_new(argv)
    if argv and argv[0] == "render":
        argv = argv[1:]
    return _main_render(argv)


def _main_new(argv):
    cmd = argv[0]
    ap = argparse.ArgumentParser(prog=f"ai_lyrics_hero {cmd}")
    ap.add_argument("--audio", required=True)
    _align_args(ap)
    if cmd == "align":
        ap.add_argument("--out", required=True, help="output .ass path")
        ap.add_argument("--config")
        ap.add_argument("--no-hero", action="store_true"); ap.add_argument("--no-images", action="store_true")
        ap.add_argument("--allow-ambiguous", action="store_true"); ap.add_argument("--title")
        ap.add_argument("--style", choices=sorted(STYLE_PRESETS))
    else:
        ap.add_argument("--out", required=True, help="output .mp4 path")
        section = ap.add_mutually_exclusive_group()
        section.add_argument("--select-best", dest="select_best", action="store_true",
                             help="scan the full song and choose the strongest continuous lyric section")
        section.add_argument("--keep-start", dest="select_best", action="store_false",
                             help="use --start as provided instead of searching the full song")
        ap.set_defaults(select_best=None)
        _common_render_args(ap)
    a = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO)
    from .workflow import make_ass, make_video, format_review
    try:
        cfg = _cfg(a)
        if cmd == "align":
            rep = make_ass(a.audio, a.out, _read(a.lyrics), a.language, _window(a), not a.no_asr, a.model, cfg,
                           romanize=a.romanize, track_title=a.track_title, artist=a.artist,
                           lyrics_lookup=not a.no_lyrics_lookup)
        else:
            W, H = _parse_size(a.size)
            auto_select = a.select_best
            if auto_select is None:
                auto_select = bool(cfg.target_duration and a.end is None and a.start == 0)
            rep = make_video(a.audio, a.out, _read(a.lyrics), cfg, a.outro, (W, H), a.fps, a.start, a.end,
                             a.encoder, a.language, _window(a), not a.no_asr, a.model,
                             progress=lambda p: print(f"\r{p:5.0%}", end="", file=sys.stderr),
                             romanize=a.romanize, auto_select=auto_select,
                             track_title=a.track_title, artist=a.artist,
                             lyrics_lookup=not a.no_lyrics_lookup)
    except Exception as e:
        print(f"\nFailed: {e}", file=sys.stderr)
        return 1
    print("\n" + format_review(rep))
    print(f"Wrote {a.out}")
    return 0


def _main_render(argv):
    ap = argparse.ArgumentParser(prog="ai_lyrics_hero", description="Render a hero-word lyric video from an ASS file")
    ap.add_argument("--ass", required=True)
    ap.add_argument("--audio")
    ap.add_argument("--out", required=True)
    _common_render_args(ap)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        W, H = _parse_size(a.size)
        render_video(a.ass, a.audio, a.out, _cfg(a), (W, H), a.fps, a.start, a.end,
                     a.outro, a.encoder, progress=lambda p: print(f"\r{p:5.0%}", end="", file=sys.stderr))
    except Exception as e:
        print(f"\nRender failed: {e}", file=sys.stderr)
        return 1
    print(f"\nWrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
