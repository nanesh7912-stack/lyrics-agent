import json
import os
import subprocess
import numpy as np
import pytest
from ai_lyrics_hero.config import HeroConfig
from ai_lyrics_hero.pipeline import render_video, probe, pick_encoder
from ai_lyrics_hero.cli import main

pytestmark = pytest.mark.slow

HERE = os.path.dirname(os.path.abspath(__file__))
ASS = os.path.join(os.path.dirname(HERE), "samples", "are_re_are_yeh_kya_hua.ass")
SIZE = (360, 640)


def sh(*a):
    subprocess.run(a, check=True, capture_output=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("media")
    audio = str(d / "tone.wav")
    sh("ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=6", audio)
    outro = str(d / "outro.mp4")
    sh("ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0xd4a017:s=1080x1920:d=1.5:r=30",
       "-f", "lavfi", "-i", "sine=frequency=880:duration=1.5", "-shortest", "-c:v", "libx264",
       "-pix_fmt", "yuv420p", "-c:a", "aac", outro)
    silent_outro = str(d / "outro_silent.mp4")
    sh("ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=640x360:d=1:r=25", "-c:v", "libx264",
       "-pix_fmt", "yuv420p", silent_outro)
    return dict(audio=audio, outro=outro, silent=silent_outro, dir=d)


def test_render_short_clip_valid_mp4_with_audio(media):
    out = str(media["dir"] / "a.mp4")
    render_video(ASS, media["audio"], out, HeroConfig(title="T"), SIZE, 24, 0.0, 4.0, encoder="libx264")
    dur, kinds = probe(out)
    assert abs(dur - 4.0) < 0.15 and "video" in kinds and "audio" in kinds
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                                      "stream=width,height", "-of", "json", out], capture_output=True, text=True).stdout)
    assert (info["streams"][0]["width"], info["streams"][0]["height"]) == SIZE


def test_outro_is_appended_and_preserved(media):
    out = str(media["dir"] / "b.mp4")
    render_video(ASS, media["audio"], out, HeroConfig(), SIZE, 24, 0.0, 3.0, outro=media["outro"], encoder="libx264")
    dur, kinds = probe(out)
    assert abs(dur - 4.5) < 0.2 and "audio" in kinds
    # last frame is the outro's gold color
    raw = subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.3", "-i", out, "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    px = np.frombuffer(raw, np.uint8).reshape(-1, 3).mean(axis=0)
    assert px[0] > 180 and px[1] > 130 and px[2] < 80
    assert not os.path.exists(out + ".main.mp4")


def test_outro_without_audio_and_different_size_works(media):
    out = str(media["dir"] / "c.mp4")
    render_video(ASS, media["audio"], out, HeroConfig(), SIZE, 24, 0.0, 2.0, outro=media["silent"], encoder="libx264")
    dur, kinds = probe(out)
    assert abs(dur - 3.0) < 0.2 and "audio" in kinds


def test_start_offset_is_respected(media):
    out = str(media["dir"] / "d.mp4")
    render_video(ASS, media["audio"], out, HeroConfig(), SIZE, 24, 3.25, 5.25, encoder="libx264")
    assert abs(probe(out)[0] - 2.0) < 0.15


def test_missing_audio_file_still_renders_silent_visuals(media):
    out = str(media["dir"] / "e.mp4")
    render_video(ASS, None, out, HeroConfig(), SIZE, 24, 0.0, 1.0, encoder="libx264")
    assert abs(probe(out)[0] - 1.0) < 0.15


def test_unreadable_audio_reports_error_not_fake_success(media):
    bad = str(media["dir"] / "bad.wav"); open(bad, "wb").write(b"not audio")
    with pytest.raises(RuntimeError):
        render_video(ASS, bad, str(media["dir"] / "f.mp4"), HeroConfig(), SIZE, 24, 0.0, 1.0, encoder="libx264")


def test_empty_lyrics_raise_clear_error(media, tmp_path):
    p = tmp_path / "empty.ass"; p.write_text("[Events]\n")
    with pytest.raises(ValueError):
        render_video(str(p), None, str(tmp_path / "x.mp4"), HeroConfig(), SIZE, 24, 0, 1)


def test_encoder_falls_back_to_software():
    assert pick_encoder("libx264") == "libx264"
    assert pick_encoder("auto") in ("libx264", "h264_qsv")


def test_cli_end_to_end_and_no_hero_flag(media):
    out = str(media["dir"] / "g.mp4")
    rc = main(["--ass", ASS, "--audio", media["audio"], "--out", out, "--end", "2", "--size", "360x640",
               "--fps", "24", "--encoder", "libx264", "--no-hero"])
    assert rc == 0 and abs(probe(out)[0] - 2.0) < 0.15
    assert main(["--ass", "/nonexistent.ass", "--out", out]) == 1
