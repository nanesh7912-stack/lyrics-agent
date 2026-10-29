import os
import subprocess
import sys
import pytest
from ai_lyrics_hero.cli import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASS = os.path.join(ROOT, "samples", "are_re_are_yeh_kya_hua.ass")


def test_help_prints_usage_for_every_command_and_exits_zero(capsys):
    for argv in (["--help"], ["make", "--help"], ["align", "--help"], ["render", "--help"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--target-duration" in out and "--outro-audio" in out and "--style" in out


def test_module_entry_point_starts(tmp_path):
    r = subprocess.run([sys.executable, "-m", "ai_lyrics_hero", "--help"], capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0 and "usage" in r.stdout.lower()


@pytest.mark.parametrize("argv", [
    ["make"],                                                     # required arguments missing
    ["make", "--audio", "a.wav", "--out", "o.mp4", "--outro-audio", "loud"],       # invalid choice
    ["render", "--ass", ASS, "--out", "o.mp4", "--style", "nonsense"],
    ["render", "--ass", ASS, "--out", "o.mp4", "--fps", "abc"],
])
def test_invalid_arguments_exit_with_status_2(argv):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2


def test_missing_input_files_give_helpful_errors_and_no_output(tmp_path, capsys):
    out = tmp_path / "x.mp4"
    assert main(["render", "--ass", str(tmp_path / "missing.ass"), "--out", str(out)]) == 1
    assert "Render failed" in capsys.readouterr().err
    assert main(["render", "--ass", ASS, "--audio", str(tmp_path / "nope.wav"), "--out", str(out)]) == 1
    assert "not found" in capsys.readouterr().err
    assert main(["make", "--audio", str(tmp_path / "nope.wav"), "--out", str(out), "--no-asr"]) == 1
    assert not out.exists() and not (tmp_path / "x.mp4.part.mp4").exists()


def test_impossible_duration_request_is_reported_clearly(tmp_path, capsys):
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=duration=3", str(wav)], check=True, capture_output=True, timeout=60)
    rc = main(["render", "--ass", ASS, "--audio", str(wav), "--out", str(tmp_path / "o.mp4"), "--target-duration", "5",
               "--end", "2"])
    assert rc == 1 and "either target_duration or an explicit end" in capsys.readouterr().err


def test_config_file_with_legacy_or_unknown_keys_is_accepted(tmp_path):
    import json
    from ai_lyrics_hero.cli import _cfg, _common_render_args
    import argparse
    cfgf = tmp_path / "old.json"
    cfgf.write_text(json.dumps({"hero_size": 180, "theme": "purple", "fps": 30, "title": "Old preset"}))
    ap = argparse.ArgumentParser(); _common_render_args(ap)
    a = ap.parse_args(["--config", str(cfgf), "--seed", "5", "--outro-audio", "mix"])
    c = _cfg(a)
    assert c.hero_size == 180 and c.title == "Old preset" and c.seed == 5 and c.outro_audio == "mix"
