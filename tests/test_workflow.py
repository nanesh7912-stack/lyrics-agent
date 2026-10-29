import json
import os
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from ai_lyrics_hero import workflow, asr as asr_mod
from ai_lyrics_hero.asr import AsrWord
from ai_lyrics_hero.ass_parser import parse_ass
from ai_lyrics_hero.config import HeroConfig
from ai_lyrics_hero.cli import main
from ai_lyrics_hero.pipeline import probe
from ai_lyrics_hero.renderer import FrameRenderer
from ai_lyrics_hero.hero import select_heroes
from ai_lyrics_hero.audio import Levels
from helpers import ev


@pytest.fixture(scope="module")
def audio(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("a") / "t.wav")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=10", p], check=True, capture_output=True)
    return p


LRC = "[00:00.50]Blue eyes tonight\n[00:03.50]Teri aankhen\n[00:06.50]Kar de mujhe pyaar\n"


def test_lrc_makes_ass_and_review_files(tmp_path, audio):
    out = str(tmp_path / "s.ass")
    rep = workflow.make_ass(audio, out, LRC, use_asr=False)
    assert rep["method"] == "lrc" and os.path.exists(out)
    assert os.path.exists(str(tmp_path / "s.review.json")) and os.path.exists(str(tmp_path / "s.review.txt"))
    assert [e.text for e in parse_ass(out)] == ["Blue eyes tonight", "Teri aankhen", "Kar de mujhe pyaar"]
    imgs = {i["image"] for i in rep["images"]}
    assert "blue_eyes" in imgs and "love" in imgs and "car" not in imgs        # 'Kar' never becomes a car


def test_plain_lyrics_without_asr_is_flagged_as_estimate(tmp_path, audio, monkeypatch):
    monkeypatch.setattr(asr_mod, "available", lambda: False)
    rep = workflow.make_ass(audio, str(tmp_path / "e.ass"), "one two three\nfour five six")
    assert rep["method"] == "estimated" and rep["lines_needing_review"] == [0, 1]
    assert any("estimated" in w or "estimate" in w for w in rep["warnings"])


def test_no_lyrics_and_no_asr_gives_clear_error(tmp_path, audio, monkeypatch):
    monkeypatch.setattr(asr_mod, "available", lambda: False)
    with pytest.raises(RuntimeError, match="faster-whisper"):
        workflow.make_ass(audio, str(tmp_path / "x.ass"), None)


def test_asr_path_keeps_exact_lyrics_spelling(tmp_path, audio, monkeypatch):
    monkeypatch.setattr(asr_mod, "available", lambda: True)
    monkeypatch.setattr(asr_mod, "transcribe", lambda *a, **k: [
        AsrWord("blu", 1.0, 1.4), AsrWord("ais", 1.4, 1.9), AsrWord("tunight", 1.9, 2.6),
        AsrWord("teri", 4.0, 4.3), AsrWord("ankhen", 4.3, 4.9)])
    rep = workflow.make_ass(audio, str(tmp_path / "a.ass"), "Blue eyes tonight\nTeri aankhen")
    evs = parse_ass(str(tmp_path / "a.ass"))
    assert evs[0].text == "Blue eyes tonight" and evs[1].text == "Teri aankhen"       # spelling from the user
    assert rep["method"] == "asr" and abs(evs[1].start - 4.0) < 0.05


def test_audio_only_draft_is_flagged_for_review(tmp_path, audio, monkeypatch):
    monkeypatch.setattr(asr_mod, "available", lambda: True)
    monkeypatch.setattr(asr_mod, "transcribe", lambda *a, **k: [
        AsrWord("hello", 1.0, 1.4), AsrWord("world", 1.5, 2.0), AsrWord("again", 4.0, 4.5)])
    rep = workflow.make_ass(audio, str(tmp_path / "d.ass"), None)
    assert rep["lines_needing_review"] and any("DRAFT" in w for w in rep["warnings"])


def test_best_lyric_window_finds_dense_repeated_section_across_full_song():
    lines = [{"text": "quiet intro", "start": 8, "end": 12, "confidence": 0.7,
              "words": [{"text": "quiet", "start": 8, "end": 10}, {"text": "intro", "start": 10, "end": 12}]}]
    for index in range(5):
        start = 50 + index * 4
        lines.append({"text": "dil ne dil ko pukara", "start": start, "end": start + 4, "confidence": 0.95,
                      "words": [{"text": word, "start": start + offset, "end": start + offset + 1}
                                for offset, word in enumerate(("dil", "ne", "dil", "ko"))]})
    picked = workflow.select_best_lyric_window(lines, 100, 20)
    assert picked["start"] == 50
    assert picked["end"] == 70
    assert picked["repeated_phrase_hits"] >= 5
    assert picked["lyric_coverage"] == 1


def test_make_video_uses_and_reports_best_full_song_window(tmp_path, monkeypatch):
    lines = [{"text": "quiet intro", "start": 8, "end": 12, "confidence": 0.7,
              "words": [{"text": "quiet", "start": 8, "end": 10}, {"text": "intro", "start": 10, "end": 12}]}]
    for index in range(5):
        start = 50 + index * 4
        lines.append({"text": "dil ne dil ko pukara", "start": start, "end": start + 4, "confidence": 0.95,
                      "words": [{"text": word, "start": start + offset, "end": start + offset + 1}
                                for offset, word in enumerate(("dil", "ne", "dil", "ko"))]})
    monkeypatch.setattr(workflow, "make_ass", lambda *args, **kwargs: {
        "method": "asr", "audio_duration": 100.0, "warnings": [], "auto_lyrics": None,
        "lines": [{**line, "needs_review": False, "notes": []} for line in lines], "images": [], "image_notes": [],
    })
    seen = {}

    def fake_render(ass, audio, out, cfg, size, fps, start, end, outro, encoder, progress=None):
        seen.update(start=start, end=end, target=cfg.target_duration)
        return SimpleNamespace(path=out, encoder="test", plan=SimpleNamespace(
            main=20.0, outro=0.0, final=20.0, total_frames=600), measured={}, warnings=[], direction=None, music=None)

    monkeypatch.setattr(workflow, "render_video", fake_render)
    out = str(tmp_path / "best.mp4")
    report = workflow.make_video("song.mp3", out, cfg=HeroConfig(target_duration=20), auto_select=True)
    assert seen == {"start": 50, "end": 70, "target": 0.0}
    assert report["source_window"] == {"start": 50, "end": 70}
    review = (tmp_path / "best.review.txt").read_text(encoding="utf-8")
    assert "selected from the full 100.00s song" in review


def test_asr_failure_falls_back_with_warning(tmp_path, audio, monkeypatch):
    monkeypatch.setattr(asr_mod, "available", lambda: True)
    def boom(*a, **k): raise OSError("model download blocked")
    monkeypatch.setattr(asr_mod, "transcribe", boom)
    rep = workflow.make_ass(audio, str(tmp_path / "f.ass"), "one two\nthree four")
    assert rep["method"] == "estimated" and any("model download blocked" in w for w in rep["warnings"])


@pytest.mark.slow
def test_make_video_end_to_end_with_images(tmp_path, audio):
    out = str(tmp_path / "v.mp4")
    rep = workflow.make_video(audio, out, LRC, HeroConfig(title="T"), size=(360, 640), fps=24, end=8.0,
                              encoder="libx264", use_asr=False)
    dur, kinds = probe(out)
    assert abs(dur - 8.0) < 0.2 and "audio" in kinds and rep["images"]


@pytest.mark.slow
def test_cli_make_and_align(tmp_path, audio):
    lrc = tmp_path / "l.lrc"; lrc.write_text(LRC, encoding="utf-8")
    assert main(["align", "--audio", audio, "--lyrics", str(lrc), "--out", str(tmp_path / "c.ass"), "--no-asr"]) == 0
    out = str(tmp_path / "c.mp4")
    assert main(["make", "--audio", audio, "--lyrics", str(lrc), "--out", out, "--no-asr", "--end", "5",
                 "--size", "360x640", "--fps", "24", "--encoder", "libx264", "--no-images"]) == 0
    assert abs(probe(out)[0] - 5.0) < 0.2
    assert main(["make", "--audio", audio, "--out", str(tmp_path / "n.mp4"), "--no-asr"]) == 1


# ---- image rendering ----
W, H = 540, 960


def rend(text, cfg=None, start=0.0, end=4.0):
    cfg = cfg or HeroConfig()
    evs = select_heroes([ev(text, start, end)], cfg)
    return FrameRenderer(cfg, evs, W, H, Levels(None), title="")


def colorful(frame, zone):
    a = np.asarray(frame, dtype=np.int16)[zone]
    return ((a.max(axis=2) - a.min(axis=2)) > 90).sum()           # saturated pixels = emoji, not text/background


ZONE = (slice(int(H * 0.55), int(H * 0.78)), slice(0, W))


def test_image_appears_only_after_its_word_and_below_the_text():
    r = rend("my red car tonight")
    cue = r.cues[0]
    assert cue.entry_id == "car"
    assert colorful(r.render(cue.time - 0.05), ZONE) == 0
    assert colorful(r.render(cue.time + 0.8), ZONE) > 800


def test_image_disabled_or_no_match_shows_nothing():
    assert colorful(rend("my red car tonight", HeroConfig(images=False)).render(2.5), ZONE) == 0
    r = rend("nothing matches here")
    assert not r.cues and colorful(r.render(2.5), ZONE) == 0


def test_image_does_not_cover_text_and_stays_above_visualizer():
    r = rend("my red car tonight")
    f = r.render(r.cues[0].time + 0.8)
    top = colorful(f, (slice(int(H * 0.30), int(H * 0.52)), slice(0, W)))
    a = np.asarray(f, dtype=np.int16)[int(H * 0.79):]
    car_red_in_viz = ((a[..., 0] > 180) & (a[..., 1] < 70) & (a[..., 2] < 70)).sum()   # bars are blue, car is red
    assert top == 0 and car_red_in_viz == 0


def test_image_fades_out_with_the_line():
    r = rend("my red car tonight")
    assert colorful(r.render(3.995), ZONE) < colorful(r.render(3.0), ZONE) * 0.2


def test_user_icon_png_is_used(tmp_path):
    from PIL import Image
    Image.new("RGBA", (200, 200), (0, 255, 0, 255)).save(tmp_path / "car.png")
    r = rend("my red car tonight", HeroConfig(icons_dir=str(tmp_path)))
    a = np.asarray(r.render(r.cues[0].time + 0.8), dtype=np.int16)[ZONE]
    assert ((a[..., 1] > 200) & (a[..., 0] < 60) & (a[..., 2] < 60)).sum() > 3000


# ---- full workflow reporting, CLI duration flags, frame sanity ----
@pytest.mark.slow
def test_make_video_report_contains_render_plan_music_and_visual_direction(tmp_path, audio):
    out = str(tmp_path / "r.mp4")
    rep = workflow.make_video(audio, out, LRC, HeroConfig(), size=(360, 640), fps=24, end=6.0, encoder="libx264", use_asr=False)
    r = rep["render"]
    assert r["final_seconds"] == 6.0 and r["frames"] == 144 and r["measured"]["video_frames"] == 144
    assert len(rep["visual_plan"]) == 3 and {"scene", "typography", "composition"} <= set(rep["visual_plan"][0])
    assert "music" in rep and rep["semantic_source"].startswith("local")
    txt = open(str(tmp_path / "r.review.txt"), encoding="utf-8").read()
    assert "Verified: video" in txt and "Visual plan" in txt


@pytest.mark.slow
def test_cli_target_duration_outro_and_audio_policy_flags(tmp_path, audio):
    outro = str(tmp_path / "o.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0xd4a017:s=1080x1920:d=2:r=30", "-f", "lavfi", "-i",
                    "sine=frequency=880:duration=2", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", outro],
                   check=True, capture_output=True)
    lrc = tmp_path / "l.lrc"; lrc.write_text(LRC, encoding="utf-8")
    out = str(tmp_path / "c.mp4")
    rc = main(["make", "--audio", audio, "--lyrics", str(lrc), "--out", out, "--no-asr", "--size", "360x640", "--fps", "24",
               "--encoder", "libx264", "--outro", outro, "--target-duration", "7", "--outro-audio", "silent", "--seed", "4",
               "--typography", "slide", "--composition", "offset", "--transitions", "zoom"])
    assert rc == 0 and abs(probe(out)[0] - 7.0) < 0.1
    bad = main(["make", "--audio", audio, "--lyrics", str(lrc), "--out", str(tmp_path / "bad.mp4"), "--no-asr",
                "--outro", outro, "--target-duration", "1"])
    assert bad == 1 and not os.path.exists(str(tmp_path / "bad.mp4"))


@pytest.mark.slow
def test_rendered_video_has_no_black_or_blank_frames_in_the_lyric_part(tmp_path, audio):
    out = str(tmp_path / "f.mp4")
    workflow.make_video(audio, out, LRC, HeroConfig(background="scenes", visualizer=False), size=(270, 480), fps=12,
                        end=8.0, encoder="libx264", use_asr=False)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-vf", "fps=2,scale=54:96", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 96, 54)
    assert len(frames) >= 15 and all(f.mean() > 8 for f in frames)             # never an all-black frame
    assert max(np.abs(a.astype(int) - b.astype(int)).mean() for a, b in zip(frames, frames[1:])) > 0.5   # not frozen


@pytest.mark.slow
def test_lyrics_appear_only_inside_their_time_range(tmp_path, audio):
    out = str(tmp_path / "t.mp4")
    workflow.make_video(audio, out, "[00:02.00]Hello dear world\n[00:07.00]x y z", HeroConfig(images=False, visualizer=False),
                        size=(270, 480), fps=12, end=5.0, encoder="libx264", use_asr=False)
    def bright(t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", out, "-frames:v", "1", "-vf", "scale=135:240",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
        a = np.frombuffer(raw, np.uint8)
        return int((a > 215).sum())
    assert bright(0.8) < 20 and bright(3.6) > 150


@pytest.mark.slow
def test_paths_with_spaces_and_non_ascii_characters_work(tmp_path):
    import shutil
    d = tmp_path / "my songs & clips \u0906"                         # spaces, ampersand, Devanagari
    d.mkdir()
    aud = d / "tone file.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=4", str(aud)], check=True,
                   capture_output=True, timeout=60)
    lrc = d / "my lyrics.lrc"; lrc.write_text(LRC, encoding="utf-8")
    out = d / "out put.mp4"
    assert main(["make", "--audio", str(aud), "--lyrics", str(lrc), "--out", str(out), "--no-asr", "--end", "3",
                 "--size", "270x480", "--fps", "12", "--encoder", "libx264"]) == 0
    assert out.exists() and (d / "out put.review.txt").exists() and abs(probe(str(out))[0] - 3.0) < 0.1
