"""End-to-end rendering: ASS + audio -> verified MP4 (optional outro with explicit audio policy).

Duration model (all quantized to whole frames):
    main_duration  = target_duration - outro_duration      (when a target is given)
    final_duration = main_duration + outro_duration
The video and the audio of the final file are both built to `final_duration` and checked with ffprobe.
"""
import functools
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .ass_parser import parse_ass
from .audio import load_audio, Levels
from .config import HeroConfig, OUTRO_AUDIO_MODES
from .assets import AssetLibrary
from .hero import select_heroes
from .music import analyze
from .renderer import FrameRenderer

log = logging.getLogger("ai_lyrics_hero")
AUDIO_TOLERANCE_S = 0.05      # ~2 AAC frames (1024 samples = 21 ms at 48 kHz): covers encoder priming/padding only, never a systematic shortfall


class RenderError(RuntimeError):
    pass


class VerificationError(RenderError):
    pass


# ------------------------------------------------------------------ probing
PROBE_TIMEOUT_S = 60


def _run_probe(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                              "format=duration:stream=index,codec_type,duration,nb_frames,width,height,r_frame_rate",
                              "-of", "json", str(path)], capture_output=True, text=True, check=True,
                             timeout=PROBE_TIMEOUT_S).stdout
    except FileNotFoundError:
        raise RenderError("ffprobe not found on PATH (install FFmpeg)")
    except subprocess.TimeoutExpired:
        raise RenderError(f"ffprobe timed out reading '{path}'")
    except subprocess.CalledProcessError as e:
        raise RenderError(f"ffprobe could not read '{path}': {(e.stderr or '').strip()[-300:]}")
    return json.loads(out)


@functools.lru_cache(maxsize=32)
def _probe_cached(path, mtime, size):
    return _run_probe(path)


def probe_input(path):
    """Cached probe for INPUT files (keyed by path+mtime+size, so edits are noticed). Not used for outputs."""
    st = os.stat(path)
    return _probe_cached(os.path.abspath(str(path)), st.st_mtime_ns, st.st_size)


def probe(path):
    """(format_duration_seconds, [stream codec types])."""
    j = _run_probe(path)
    try:
        dur = float(j["format"]["duration"])
    except (KeyError, ValueError):
        raise RenderError(f"'{path}' has no readable duration")
    return dur, [s["codec_type"] for s in j.get("streams", [])]


def probe_streams(path):
    j = _run_probe(path)
    return j.get("format", {}), j.get("streams", [])


def _frac(s):
    try:
        a, b = s.split("/")
        return float(a) / float(b)
    except (ValueError, ZeroDivisionError, AttributeError):
        return 0.0


# ------------------------------------------------------------------ encoder selection
@functools.lru_cache(maxsize=4)
def _qsv_usable():
    """True only if this FFmpeg lists h264_qsv AND a real 0.2 s test encode succeeds (cached per process)."""
    try:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
    except FileNotFoundError:
        raise RenderError("ffmpeg not found on PATH (install FFmpeg)")
    except subprocess.TimeoutExpired:
        return False
    if "h264_qsv" not in enc:
        return None                      # not even listed
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                            "-c:v", "h264_qsv", "-f", "null", "-"], capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def pick_encoder(preferred="auto"):
    """Return an ffmpeg video encoder; Intel QSV only if a real test encode succeeds, else libx264."""
    if preferred == "libx264":
        return "libx264"
    usable = _qsv_usable()
    if usable:
        return "h264_qsv"
    if usable is False:
        log.warning("h264_qsv listed but not usable here; using libx264")
    elif preferred == "h264_qsv":
        log.warning("h264_qsv not available in this FFmpeg build; using libx264")
    return "libx264"


def _enc_args(enc):
    if enc == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", "21", "-pix_fmt", "nv12"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]


@functools.lru_cache(maxsize=1)
def _amix_has_normalize():
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-h", "filter=amix"], capture_output=True, text=True,
                             timeout=30).stdout
        return "normalize" in out
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ------------------------------------------------------------------ duration planning
@dataclass
class DurationPlan:
    fps: int
    main_frames: int
    outro_frames: int
    start: float = 0.0
    audio_available: Optional[float] = None       # seconds of song available from `start`
    warnings: List[str] = field(default_factory=list)

    @property
    def main(self):
        return self.main_frames / self.fps

    @property
    def outro(self):
        return self.outro_frames / self.fps

    @property
    def total_frames(self):
        return self.main_frames + self.outro_frames

    @property
    def final(self):
        return self.total_frames / self.fps


def plan_durations(audio_duration, outro_actual, fps=30, start=0.0, end=None, target=0.0, outro_duration=0.0,
                   fallback_main=None):
    """Compute the main/outro/final durations. Raises ValueError with a helpful message when impossible.

    audio_duration  length of the song file (None when no audio is used)
    outro_actual    length of the outro file (None/0 when there is no outro)
    end             explicit end time in the song for the main section (preview ranges)
    target          wanted final length INCLUDING the outro
    outro_duration  seconds of the outro to use (0 = all of it)
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if start < 0:
        raise ValueError("start must be >= 0")
    if outro_duration is not None and outro_duration < 0:
        raise ValueError("outro_duration must be >= 0")
    warnings = []
    has_outro = bool(outro_actual and outro_actual > 0)
    o_len = 0.0
    if has_outro:
        o_len = outro_actual if not outro_duration else outro_duration
        if o_len > outro_actual + 1.0 / fps:
            raise ValueError(f"outro_duration {o_len:.2f}s is longer than the outro file ({outro_actual:.2f}s)")
        o_len = min(o_len, outro_actual)
    avail = None if audio_duration is None else audio_duration - start
    if avail is not None and avail <= 0:
        raise ValueError(f"start ({start}s) is beyond the end of the audio ({audio_duration:.2f}s)")
    if target and end is not None:
        raise ValueError("give either target_duration or an explicit end, not both")
    if target:
        main = target - o_len
        if main < 0.5:
            raise ValueError(f"target_duration {target:.2f}s leaves only {main:.2f}s for the lyric video after the "
                             f"{o_len:.2f}s outro")
    elif end is not None:
        main = end - start
    elif avail is not None:
        main = avail - o_len if has_outro else avail            # reserve the outro inside the song length
        if main < 0.5:
            raise ValueError(f"audio ({avail:.2f}s) is too short to fit a {o_len:.2f}s outro")
    elif fallback_main:
        main = fallback_main
    else:
        raise ValueError("cannot determine the video length: give audio, target_duration or end")
    if main <= 0:
        raise ValueError(f"main section length must be positive (got {main:.2f}s)")
    plan = DurationPlan(fps=int(fps), main_frames=max(1, int(round(main * fps))),
                        outro_frames=int(round(o_len * fps)), start=start, audio_available=avail, warnings=warnings)
    if avail is not None and avail + 1e-6 < plan.final:
        warnings.append(f"the song has {avail:.2f}s after the start point but the video is {plan.final:.2f}s long; "
                        f"the remaining {plan.final - avail:.2f}s will be silent (music is never looped or invented)")
    return plan


# ------------------------------------------------------------------ audio graph
def build_audio_graph(plan, policy, crossfade, have_song, have_outro_audio, song_idx, outro_idx, amix_normalize=True):
    """Return filter-complex text that produces [a] with exactly plan.final seconds of audio."""
    if policy not in OUTRO_AUDIO_MODES:
        raise ValueError(f"unknown outro audio policy '{policy}'")
    st, main, outro, final = plan.start, plan.main, plan.outro, plan.final
    fmt = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
    use_outro_audio = policy in ("outro", "mix") and plan.outro_frames > 0 and have_outro_audio
    if policy in ("outro", "mix") and plan.outro_frames > 0 and not have_outro_audio:
        plan.warnings.append(f"outro_audio='{policy}' requested but the outro has no audio; the outro will be silent")
    song_in = f"[{song_idx}:a]{fmt}," if have_song else f"[{song_idx}:a]"
    def song_seg(length, tail=""):
        if have_song:
            return f"{song_in}atrim=start={st:.4f}:end={st + length:.4f},asetpts=PTS-STARTPTS{tail}"
        return f"{song_in}atrim=end={length:.4f},asetpts=PTS-STARTPTS{tail}"
    if plan.outro_frames == 0:
        return song_seg(main, f",apad=whole_dur={main:.4f},atrim=end={main:.4f}[a]")
    if policy == "continue" or (policy in ("outro", "mix") and not use_outro_audio and policy != "silent"):
        if policy == "continue":
            return song_seg(final, f",apad=whole_dur={final:.4f},atrim=end={final:.4f}[a]")
        # requested outro/mix but no outro audio: keep the song going to the end of the main part, then silence
        return song_seg(main, f",apad=whole_dur={final:.4f},atrim=end={final:.4f}[a]")
    if policy == "silent":
        return song_seg(main, f",apad=whole_dur={final:.4f},atrim=end={final:.4f}[a]")
    o_seg = f"[{outro_idx}:a]{fmt},atrim=end={outro:.4f},asetpts=PTS-STARTPTS"
    if policy == "outro":
        return (song_seg(main, f",apad=whole_dur={main:.4f},atrim=end={main:.4f}[s];") +
                f"{o_seg},apad=whole_dur={outro:.4f},atrim=end={outro:.4f}[o];"
                f"[s][o]concat=n=2:v=0:a=1,atrim=end={final:.4f}[a]")
    # mix: song fades out across the boundary while the outro audio fades in
    xf = max(0.01, min(crossfade, outro, main))
    norm = ":normalize=0" if amix_normalize else ""
    post = "" if amix_normalize else ",volume=2"
    return (song_seg(main + xf, f",afade=t=out:st={main:.4f}:d={xf:.4f},apad=whole_dur={final:.4f}[s];") +
            f"{o_seg},afade=t=in:st=0:d={xf:.4f},adelay={int(round(main * 1000))}:all=1,apad=whole_dur={final:.4f}[o];"
            f"[s][o]amix=inputs=2:duration=longest:dropout_transition=0{norm}{post},atrim=end={final:.4f}[a]")


# ------------------------------------------------------------------ verification
def verify_output(path, plan, size, fps, expect_audio=True, decode=True):
    """Check the finished file with ffprobe (+ a full decode). Returns measurements; raises on mismatch."""
    fmt, streams = probe_streams(path)
    v = [s for s in streams if s["codec_type"] == "video"]
    a = [s for s in streams if s["codec_type"] == "audio"]
    problems = []
    meas = {"format_duration": float(fmt.get("duration", 0))}
    if not v:
        problems.append("no video stream")
    else:
        v = v[0]
        vd = float(v.get("duration", fmt.get("duration", 0)))
        meas.update(video_duration=vd, width=v.get("width"), height=v.get("height"),
                    fps=_frac(v.get("r_frame_rate", "0/1")), video_frames=int(v.get("nb_frames", 0) or 0))
        if (v.get("width"), v.get("height")) != tuple(size):
            problems.append(f"resolution {v.get('width')}x{v.get('height')} != {size[0]}x{size[1]}")
        if abs(meas["fps"] - fps) > 0.01:
            problems.append(f"fps {meas['fps']:.3f} != {fps}")
        if abs(vd - plan.final) > 0.5 / fps + 1e-3:
            problems.append(f"video stream is {vd:.3f}s, expected {plan.final:.3f}s")
        if meas["video_frames"] and abs(meas["video_frames"] - plan.total_frames) > 1:
            problems.append(f"video has {meas['video_frames']} frames, expected {plan.total_frames}")
    if expect_audio:
        if not a:
            problems.append("no audio stream")
        else:
            ad = float(a[0].get("duration", fmt.get("duration", 0)))
            meas["audio_duration"] = ad
            if abs(ad - plan.final) > AUDIO_TOLERANCE_S:
                problems.append(f"audio stream is {ad:.3f}s, expected {plan.final:.3f}s (tolerance {AUDIO_TOLERANCE_S}s)")
    if decode and not problems:
        try:
            r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True,
                               text=True, timeout=max(60, 10 * plan.final))
        except subprocess.TimeoutExpired:
            raise VerificationError("output verification failed: decode check timed out")
        if r.returncode != 0 or r.stderr.strip():
            problems.append(f"decode check failed: {r.stderr.strip()[-300:]}")
    if problems:
        raise VerificationError("output verification failed: " + "; ".join(problems))
    return meas


# ------------------------------------------------------------------ rendering
@dataclass
class RenderResult:
    path: str
    plan: DurationPlan
    encoder: str
    measured: dict
    warnings: List[str]
    direction: object = None
    music: object = None


FFMPEG_IDLE_TIMEOUT_S = float(os.environ.get("AI_LYRICS_FFMPEG_IDLE_TIMEOUT", "180"))


def _run_ffmpeg_with_frames(cmd, renderer, plan, t0, progress):
    """Feed frames to ffmpeg; stderr goes to a temp file (no pipe deadlock).

    A watchdog kills ffmpeg if it stops accepting frames for FFMPEG_IDLE_TIMEOUT_S seconds (a hung encoder must not hang
    the application forever); time spent drawing frames does not count. Returns (rc, stderr_text, broken_pipe)."""
    with tempfile.TemporaryFile() as errf:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
        state = {"last": time.monotonic(), "stop": False, "killed": False}

        def watch():
            while not state["stop"]:
                time.sleep(1.0)
                if time.monotonic() - state["last"] > FFMPEG_IDLE_TIMEOUT_S and not state["stop"]:
                    state["killed"] = True
                    p.kill()
                    return
        wd = threading.Thread(target=watch, daemon=True)
        wd.start()
        broken = False
        try:
            for i in range(plan.main_frames):
                frame = renderer.render(t0 + i / plan.fps).tobytes()
                state["last"] = time.monotonic()                      # drawing is ours; only the write is watched
                p.stdin.write(frame)
                state["last"] = time.monotonic()
                if progress and i % 30 == 0:
                    progress(i / plan.main_frames)
            p.stdin.close()
        except (BrokenPipeError, OSError):
            broken = True
        except BaseException:
            state["stop"] = True
            p.kill(); p.wait()
            raise
        try:
            rc = p.wait(timeout=max(120.0, 10.0 * plan.final))
        except subprocess.TimeoutExpired:
            state["killed"] = True
            p.kill(); rc = p.wait()
        state["stop"] = True
        errf.seek(0)
        err = errf.read().decode(errors="replace")
        if state["killed"]:
            err = "ffmpeg stopped responding and was terminated by the watchdog. " + err
            rc = rc or 1
    return rc, err, broken


def _info(path):
    j = probe_input(path)
    try:
        dur = float(j["format"]["duration"])
    except (KeyError, ValueError):
        raise RenderError(f"'{path}' has no readable duration")
    return {"duration": dur, "kinds": [x["codec_type"] for x in j.get("streams", [])]}


def render_video(ass, audio, out, cfg=None, size=(1080, 1920), fps=30, start=0.0, end=None, outro=None,
                 encoder="auto", title=None, progress=None, target_duration=None, outro_duration=None,
                 outro_audio=None, crossfade=None, verify=None):
    cfg = cfg or HeroConfig()
    W, H = size
    target = cfg.target_duration if target_duration is None else target_duration
    o_dur = cfg.outro_duration if outro_duration is None else outro_duration
    policy = cfg.outro_audio if outro_audio is None else outro_audio
    xfade = cfg.crossfade_s if crossfade is None else crossfade
    do_verify = cfg.verify_output if verify is None else verify
    if policy not in OUTRO_AUDIO_MODES:
        raise ValueError(f"outro_audio must be one of {OUTRO_AUDIO_MODES}")

    events = parse_ass(ass)
    if not events:
        raise ValueError("no lyric events found in the ASS input")
    for label, path in (("audio", audio), ("outro", outro)):
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")
    output_path = os.path.normcase(os.path.abspath(os.fspath(out)))
    for label, path in (("ASS", ass), ("audio", audio), ("outro", outro)):
        if not path:
            continue
        try:
            input_path = os.fspath(path)
            if not os.path.isfile(input_path):
                continue
            same_path = output_path == os.path.normcase(os.path.abspath(input_path))
            if not same_path and os.path.exists(out):
                same_path = os.path.samefile(out, input_path)
        except (OSError, TypeError, ValueError):
            same_path = False
        if same_path:
            raise ValueError(f"output file must not replace the {label} input: {path}")
    a_dur = None
    if audio:
        ainfo = _info(audio)
        a_dur = ainfo["duration"]
        if "audio" not in ainfo["kinds"]:
            raise RenderError(f"'{audio}' contains no audio stream")
    o_dur_actual, o_kinds = (None, [])
    if outro:
        oinfo = _info(outro)
        o_dur_actual, o_kinds = oinfo["duration"], oinfo["kinds"]
        if "video" not in o_kinds:
            raise RenderError(f"outro '{outro}' contains no video stream")
    plan = plan_durations(a_dur, o_dur_actual, fps, start, end, target, o_dur,
                          fallback_main=(max(e.end for e in events) + 1.0 if not audio else None))
    samples = load_audio(audio, start, plan.main) if audio else None
    if audio and samples is None:
        raise RenderError(f"could not decode audio from '{audio}' (corrupt or unsupported file?)")
    music = analyze(samples, offset=start)
    for note in music.notes:
        plan.warnings.append("music analysis: " + note)
    select_heroes(events, cfg, music)
    assets = AssetLibrary(cfg.asset_dirs, size) if cfg.asset_dirs else None
    if assets is not None:
        plan.warnings.extend(assets.warnings)
    out = str(out)
    parent = os.path.dirname(os.path.abspath(out))
    os.makedirs(parent, exist_ok=True)
    part = out + ".part.mp4"
    try:
        enc = pick_encoder(encoder)
        attempts = [enc] + (["libx264"] if enc != "libx264" else [])
        last_err = ""
        for attempt, e in enumerate(attempts):
            renderer = FrameRenderer(cfg, events, W, H, Levels(samples, offset=start), title, music=music,
                                     assets=assets, total_end=start + plan.main)
            cmd = _build_command(plan, audio, outro, part, size, e, policy, xfade, start, o_kinds)
            rc, err, broken = _run_ffmpeg_with_frames(cmd, renderer, plan, start, progress)
            if rc == 0:
                break
            last_err = err
            log.warning("ffmpeg failed with encoder %s: %s", e, err[-300:])
            if os.path.exists(part):
                os.remove(part)
        else:
            raise RenderError(f"ffmpeg failed: {last_err[-800:]}")
        measured = verify_output(part, plan, size, fps, True, decode=True) if do_verify else {}
        os.replace(part, out)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    for w in plan.warnings:
        log.warning(w)
    res = RenderResult(out, plan, e, measured, list(plan.warnings))
    res.direction = renderer.direction
    res.music = music
    return res


def _build_command(plan, audio, outro, part, size, enc, policy, xfade, start, o_kinds=None):
    W, H = size
    fps = plan.fps
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-"]
    idx = 1
    outro_idx = song_idx = None
    have_outro_audio = False
    if outro:
        cmd += ["-i", str(outro)]
        outro_idx = idx; idx += 1
        have_outro_audio = "audio" in (o_kinds if o_kinds is not None else _info(outro)["kinds"])
    if audio:
        cmd += ["-i", str(audio)]
    else:
        cmd += ["-f", "lavfi", "-t", f"{plan.final:.4f}", "-i", "anullsrc=r=48000:cl=stereo"]
    song_idx = idx
    vf = f"[0:v]fps={fps},format=yuv420p,setsar=1[v0];"
    if plan.outro_frames:
        vf += (f"[{outro_idx}:v]trim=duration={plan.outro:.4f},setpts=PTS-STARTPTS,"
               f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
               f"fps={fps},format=yuv420p,setsar=1[v1];[v0][v1]concat=n=2:v=1:a=0[vc];")
    else:
        vf += "[v0]null[vc];"
    # the closing fps filter restores the frame rate the tpad/trim/setpts chain loses (FFmpeg >= 7.1 then gives the last
    # frame no duration and the video stream comes out one frame short)
    vf += (f"[vc]tpad=stop_mode=clone:stop_duration=1,trim=end_frame={plan.total_frames},setpts=PTS-STARTPTS,"
           f"fps={fps}[v]")
    af = build_audio_graph(plan, policy, xfade, bool(audio), have_outro_audio, song_idx, outro_idx,
                           _amix_has_normalize() if policy == "mix" else True)
    cmd += ["-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]"] + _enc_args(enc)
    # Root cause of "audio stream is 7.914s, expected 8.000s" on FFmpeg 7.1 (verified by ablation on a real 7.1.1 build):
    #  1. the filter chain lost the frame rate, so the muxer gave the last video frame no duration (video = (N-1)/fps);
    #  2. `-frames:v N` then ended the whole output at that last frame, cutting the audio there too.
    # Fix: state the output frame rate explicitly (-r + closing fps filter) and do NOT use -frames:v: the graph itself
    # emits exactly total_frames frames (trim=end_frame) and the audio graph ends at exactly `final` seconds.
    cmd += ["-c:a", "aac", "-b:a", "192k", "-r", str(fps), "-movflags", "+faststart", "-f", "mp4", part]
    return cmd
