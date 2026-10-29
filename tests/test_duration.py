import os
import subprocess
import numpy as np
import pytest
from ai_lyrics_hero import pipeline
from ai_lyrics_hero.config import HeroConfig
from ai_lyrics_hero.pipeline import (plan_durations, render_video, verify_output, probe_streams, RenderError,
                                     VerificationError, DurationPlan)

pytestmark = pytest.mark.slow          # renders real videos through FFmpeg (quick run: -m "not slow")

HERE = os.path.dirname(os.path.abspath(__file__))
ASS = os.path.join(os.path.dirname(HERE), "samples", "are_re_are_yeh_kya_hua.ass")
SIZE, FPS = (360, 640), 24


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("dur")
    song = str(d / "song.wav"); sh("ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=8", "-ar", "48000", song)
    short = str(d / "short.wav"); sh("ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=5", "-ar", "48000", short)
    outro = str(d / "outro.mp4")
    sh("ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0xd4a017:s=1080x1920:d=2:r=30", "-f", "lavfi",
       "-i", "sine=frequency=880:duration=2", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", outro)
    mute = str(d / "mute.mp4")
    sh("ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=640x360:d=2:r=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", mute)
    return dict(song=song, short=short, outro=outro, mute=mute, dir=d)


_PCM = {}


def pcm(path, a, b):
    """Audio of [a, b) seconds. The whole file is decoded ONCE per path (process launches are the slow part)."""
    if path not in _PCM:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                             capture_output=True, timeout=120).stdout
        _PCM[path] = np.frombuffer(raw, np.float32)
    return _PCM[path][int(a * 16000):int(b * 16000)]


def band(x, f0, width=40):
    if len(x) < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x)
    freqs = np.fft.rfftfreq(len(x), 1 / 16000)
    return float(spec[(freqs > f0 - width) & (freqs < f0 + width)].sum())


def rms(x):
    return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0


_RENDERS = {}


def go(media, name, **kw):
    """Render once per (file name); tests that need the same configuration share the result."""
    key = (str(media["dir"]), name)
    if key not in _RENDERS:
        out = str(media["dir"] / name)
        kw.setdefault("encoder", "libx264")
        res = render_video(ASS, kw.pop("audio", media["song"]), out, kw.pop("cfg", HeroConfig()), SIZE, FPS, **kw)
        _RENDERS[key] = (out, res)
    return _RENDERS[key]


def policy_render(media, policy):
    kw = dict(outro=media["outro"], target_duration=6.0, outro_audio=policy)
    if policy == "mix":
        kw["crossfade"] = 1.0
    return go(media, {"continue": "c.mp4", "outro": "o.mp4", "silent": "s.mp4", "mix": "m.mp4"}[policy], **kw)


# ------------------------------------------------------------- planning (pure)
def test_user_example_40s_target_with_65s_outro_at_30fps():
    p = plan_durations(40.0, 6.5, fps=30, target=40.0)
    assert (p.main_frames, p.outro_frames, p.total_frames) == (1005, 195, 1200)
    assert p.main == 33.5 and p.outro == 6.5 and p.final == 40.0


def test_default_reserves_the_outro_inside_the_song_length():
    p = plan_durations(40.0, 6.5, fps=30)
    assert abs(p.final - 40.0) < 1e-9 and abs(p.main - 33.5) < 1e-9


def test_no_outro_uses_whole_song_and_end_is_explicit_window():
    assert plan_durations(10.0, None, 30).final == 10.0
    p = plan_durations(40.0, 6.5, 30, start=2.0, end=12.0)
    assert abs(p.main - 10.0) < 1e-9 and abs(p.final - 16.5) < 1e-9


def test_outro_duration_trims_and_actual_length_is_used_otherwise():
    assert plan_durations(40, 6.5, 30, target=40, outro_duration=3).main == 37.0
    assert plan_durations(40, 6.566, 30, target=40).outro_frames == round(6.566 * 30)


def test_quantized_to_whole_frames():
    p = plan_durations(10.0, 1.0, 30, end=3.3333)
    assert p.main_frames == 100 and p.final == (100 + 30) / 30


@pytest.mark.parametrize("kw, msg", [
    (dict(target=5.0, outro_duration=0), "leaves only"),
    (dict(target=10, end=5), "either target_duration or an explicit end"),
    (dict(outro_duration=9.0, target=30), "longer than the outro file"),
    (dict(start=99.0), "beyond the end"),
])
def test_impossible_requests_are_rejected_with_clear_messages(kw, msg):
    with pytest.raises(ValueError, match=msg):
        plan_durations(40.0, 6.5, 30, **kw)


def test_audio_too_short_for_outro_and_unknown_length():
    with pytest.raises(ValueError, match="too short"):
        plan_durations(6.0, 6.5, 30)
    with pytest.raises(ValueError, match="cannot determine"):
        plan_durations(None, None, 30)


def test_song_shorter_than_video_warns_instead_of_looping():
    p = plan_durations(5.0, 2.0, 24, target=8.0)
    assert any("silent" in w for w in p.warnings)


def test_unknown_policy_rejected():
    p = plan_durations(8, 2, 24)
    with pytest.raises(ValueError):
        pipeline.build_audio_graph(p, "bogus", 1, True, True, 2, 1)


# ------------------------------------------------------------- real renders: durations + audio policies
def check_durations(out, res):
    fmt, streams = probe_streams(out)
    v = [s for s in streams if s["codec_type"] == "video"][0]
    a = [s for s in streams if s["codec_type"] == "audio"][0]
    assert int(v["nb_frames"]) == res.plan.total_frames
    assert abs(float(v["duration"]) - res.plan.final) < 1.5 / FPS
    assert abs(float(a["duration"]) - res.plan.final) < pipeline.AUDIO_TOLERANCE_S
    assert not os.path.exists(out + ".part.mp4")


def test_target_duration_main_plus_outro_equals_target_for_video_and_audio(media):
    out, res = go(media, "t.mp4", outro=media["outro"], target_duration=6.0)
    assert (res.plan.main, res.plan.outro, res.plan.final) == (4.0, 2.0, 6.0)
    check_durations(out, res)
    assert res.measured["video_frames"] == 144 and abs(res.measured["audio_duration"] - 6.0) < 0.08


def test_continue_policy_song_keeps_playing_under_the_outro(media):
    out, res = policy_render(media, "continue")
    check_durations(out, res)
    tail = pcm(out, 4.3, 5.7)
    assert band(tail, 300) > 5 * band(tail, 880) and rms(tail) > 0.05
    assert band(pcm(out, 3.0, 3.8), 300) > 0.01                                  # not truncated before the boundary


def test_outro_policy_uses_the_outro_audio_after_the_song(media):
    out, res = policy_render(media, "outro")
    check_durations(out, res)
    assert band(pcm(out, 4.3, 5.7), 880) > 5 * band(pcm(out, 4.3, 5.7), 300)
    assert band(pcm(out, 2.0, 3.5), 300) > 5 * band(pcm(out, 2.0, 3.5), 880)


def test_silent_policy_has_silence_under_the_outro(media):
    out, res = policy_render(media, "silent")
    check_durations(out, res)
    assert rms(pcm(out, 4.3, 5.7)) < 0.005 and rms(pcm(out, 2.0, 3.5)) > 0.05


def test_mix_policy_crossfades_song_and_outro(media):
    out, res = policy_render(media, "mix")
    check_durations(out, res)
    early, late = pcm(out, 4.1, 4.7), pcm(out, 5.3, 5.9)
    assert band(early, 300) > 0.003 and band(early, 880) > 0.003                  # both audible during the crossfade
    assert band(late, 880) > 5 * band(late, 300)                                  # outro takes over


def test_outro_without_audio_and_outro_policy_warns_and_stays_in_sync(media):
    out, res = go(media, "n.mp4", outro=media["mute"], target_duration=6.0, outro_audio="outro")
    check_durations(out, res)
    assert any("no audio" in w for w in res.warnings) and rms(pcm(out, 4.3, 5.7)) < 0.005


def test_outro_duration_trim_is_honored(media):
    out, res = go(media, "x.mp4", outro=media["outro"], end=4.0, outro_duration=1.0)
    assert res.plan.outro == 1.0 and abs(res.plan.final - 5.0) < 1e-9
    check_durations(out, res)


def test_song_shorter_than_target_pads_with_silence_and_reports_it(media):
    out, res = go(media, "p.mp4", audio=media["short"], outro=media["outro"], target_duration=8.0)
    check_durations(out, res)
    assert any("silent" in w for w in res.warnings)
    assert rms(pcm(out, 5.5, 7.5)) < 0.005 and rms(pcm(out, 1.0, 3.0)) > 0.05


def test_no_outro_whole_song_duration(media):
    out, res = go(media, "w.mp4")
    assert abs(res.plan.final - 8.0) < 1e-9
    check_durations(out, res)


# ------------------------------------------------------------- failures
def test_missing_files_and_bad_inputs(media, tmp_path):
    out = str(tmp_path / "f.mp4")
    with pytest.raises(FileNotFoundError):
        render_video(ASS, media["song"], out, HeroConfig(), SIZE, FPS, outro=str(tmp_path / "nope.mp4"))
    with pytest.raises(FileNotFoundError):
        render_video(ASS, str(tmp_path / "nope.wav"), out, HeroConfig(), SIZE, FPS)
    bad = tmp_path / "bad.wav"; bad.write_bytes(b"not audio")
    with pytest.raises(RenderError):
        render_video(ASS, str(bad), out, HeroConfig(), SIZE, FPS)
    assert not os.path.exists(out)


def test_corrupt_outro_fails_cleanly_without_partial_output(media, tmp_path):
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"\x00" * 2000)
    out = str(tmp_path / "g.mp4")
    with pytest.raises(RenderError):
        render_video(ASS, media["song"], out, HeroConfig(), SIZE, FPS, outro=str(bad), encoder="libx264")
    assert not os.path.exists(out) and not os.path.exists(out + ".part.mp4")


def test_existing_output_is_not_clobbered_by_a_failed_render(media, tmp_path):
    out = tmp_path / "keep.mp4"; out.write_bytes(b"previous good export")
    with pytest.raises(Exception):
        render_video(ASS, media["song"], str(out), HeroConfig(), SIZE, FPS, outro=str(tmp_path / "none.mp4"))
    assert out.read_bytes() == b"previous good export"


def test_verification_catches_a_video_shorter_than_the_audio(tmp_path):
    bad = str(tmp_path / "bad.mp4")
    sh("ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=360x640:d=3:r=24", "-f", "lavfi", "-i", "sine=duration=5",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", bad)       # video 3 s, audio 5 s: the reported bug
    plan = DurationPlan(fps=24, main_frames=120, outro_frames=0)
    with pytest.raises(VerificationError, match="video stream"):
        verify_output(bad, plan, SIZE, 24)


def test_hardware_encoder_failure_falls_back_to_software(media, monkeypatch):
    monkeypatch.setattr(pipeline, "pick_encoder", lambda pref="auto": "h264_qsv")   # not usable in this FFmpeg/GPU
    out = str(media["dir"] / "q.mp4")
    res = render_video(ASS, media["song"], out, HeroConfig(), SIZE, FPS, end=2.0)
    if res.encoder == "h264_qsv":
        pytest.skip("this machine really has working QSV")
    assert res.encoder == "libx264" and os.path.exists(out)


# ------------------------------------------------------------- regression: audio/video stream lengths (FFmpeg 7.x)
@pytest.mark.parametrize("fps, end", [(12, 5.0), (12, 8.0), (24, 3.4), (25, 2.0), (30, 2.5)])
def test_audio_and_video_streams_both_cover_the_whole_duration(media, fps, end):
    """Regression for 'audio stream is 7.914s, expected 8.000s': the audio must not end one video frame early.

    The tolerance here (30 ms) is stricter than the verifier's: it only allows AAC frame granularity."""
    out = str(media["dir"] / f"cov_{fps}_{end}.mp4")
    res = render_video(ASS, media["song"], out, HeroConfig(verify_output=False), SIZE, fps, end=end, encoder="libx264")
    fmt, streams = probe_streams(out)
    v = [s for s in streams if s["codec_type"] == "video"][0]
    a = [s for s in streams if s["codec_type"] == "audio"][0]
    assert float(v["r_frame_rate"].split("/")[0]) / float(v["r_frame_rate"].split("/")[1]) == fps
    assert int(v["nb_frames"]) == round(end * fps)
    assert abs(float(v["duration"]) - end) < 0.5 / fps + 1e-6
    assert abs(float(a["duration"]) - end) < 0.03, (a["duration"], end)
    assert float(a["duration"]) > float(v["duration"]) - 1.0 / fps          # never a whole video frame short


def test_audio_covers_the_outro_too_with_every_policy(media):
    for policy in ("continue", "outro", "silent", "mix"):
        out, res = policy_render(media, policy)                     # shared with the policy tests (already verified)
        _, streams = probe_streams(out)
        a = [s for s in streams if s["codec_type"] == "audio"][0]
        v = [s for s in streams if s["codec_type"] == "video"][0]
        assert abs(float(a["duration"]) - res.plan.final) < 0.03, (policy, a["duration"], res.plan.final)
        assert abs(float(v["duration"]) - res.plan.final) < 0.5 / FPS + 1e-6
        assert int(v["nb_frames"]) == res.plan.total_frames


def test_watchdog_terminates_a_hung_encoder(monkeypatch):
    import sys, time
    from PIL import Image
    monkeypatch.setattr(pipeline, "FFMPEG_IDLE_TIMEOUT_S", 2.0)
    class R:
        def render(self, t):
            return Image.new("RGB", SIZE)
    plan = DurationPlan(fps=24, main_frames=500, outro_frames=0)
    t0 = time.monotonic()
    rc, err, broken = pipeline._run_ffmpeg_with_frames([sys.executable, "-c", "import time; time.sleep(120)"], R(), plan, 0.0, None)
    assert rc != 0 and "watchdog" in err and time.monotonic() - t0 < 30


def test_encoder_and_input_probes_are_cached(media, monkeypatch):
    calls = []
    real = subprocess.run
    def counting(cmd, *a, **k):
        calls.append(cmd[0] + " " + " ".join(str(x) for x in cmd[1:3]))
        return real(cmd, *a, **k)
    pipeline._qsv_usable.cache_clear(); pipeline._probe_cached.cache_clear()
    monkeypatch.setattr(pipeline.subprocess, "run", counting)
    for _ in range(3):
        pipeline.pick_encoder("auto")
        pipeline.probe_input(media["song"])
    assert sum("-encoders" in c for c in calls) == 1 and sum(c.startswith("ffprobe") for c in calls) == 1


def test_probe_input_notices_a_changed_file(tmp_path):
    p = tmp_path / "a.wav"
    sh("ffmpeg", "-y", "-f", "lavfi", "-i", "sine=duration=1", str(p))
    d1 = float(pipeline.probe_input(str(p))["format"]["duration"])
    sh("ffmpeg", "-y", "-f", "lavfi", "-i", "sine=duration=2", str(p))
    assert float(pipeline.probe_input(str(p))["format"]["duration"]) > d1 + 0.5
