import io
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

from ai_lyrics_hero.webapp import create_app


class ImmediateExecutor:
    def submit(self, function, *args, **kwargs):
        function(*args, **kwargs)


@pytest.fixture
def app(tmp_path, monkeypatch):
    from ai_lyrics_hero import workflow
    calls = {}

    def fake_make_video(audio, out_mp4, lyrics_text=None, cfg=None, **kwargs):
        calls["auto_select"] = kwargs.get("auto_select")
        assert cfg.target_duration == 4
        Path(out_mp4).write_bytes(b"mock video")
        Path(out_mp4).with_suffix(".review.txt").write_text("Timing method: lrc\n", encoding="utf-8")
        if lyrics_text is None:
            Path(out_mp4).with_suffix(".auto.lrc").write_text("[00:00.00]hello\n", encoding="utf-8")
        progress = kwargs.get("progress")
        if progress:
            progress(0.75)
        return {"render": {"final_seconds": 4}}

    monkeypatch.setattr(workflow, "make_video", fake_make_video)
    application = create_app({
        "TESTING": True,
        "SECRET_KEY": "test-secret-key",
        "DATABASE": str(tmp_path / "db.sqlite3"),
        "DATA_DIR": str(tmp_path / "data"),
        "JOB_EXECUTOR": ImmediateExecutor(),
    })
    application.extensions["fake_video_calls"] = calls
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def csrf(client):
    return client.get("/api/session").get_json()["csrf_token"]


def test_browser_app_serves_responsive_interface_and_session_token(client):
    page = client.get("/")
    assert page.status_code == 200
    assert b"Create lyric video" in page.data
    assert b"viewport" in page.data
    assert b"Find the best lyric section across the whole song" in page.data
    session = client.get("/api/session").get_json()
    assert session["csrf_token"]
    assert session["contact_email"] == "nanesh7912@gmail.com"


def test_csrf_and_project_crud_are_session_scoped(client):
    assert client.post("/api/projects", json={"title": "Song"}).status_code == 403
    token = csrf(client)
    response = client.post("/api/projects", headers={"X-CSRF-Token": token},
                           json={"title": "My song", "settings": {"duration": 40, "style": "scene-hero"}})
    assert response.status_code == 201
    project = response.get_json()["project"]
    assert project["settings"]["duration"] == 40
    updated = client.put(f"/api/projects/{project['id']}", headers={"X-CSRF-Token": token},
                         json={"title": "Updated song", "settings": {"duration": 30}})
    assert updated.get_json()["project"]["title"] == "Updated song"
    assert client.get(f"/api/projects/{project['id']}").get_json()["project"]["settings"]["duration"] == 30
    other_browser = client.application.test_client()
    csrf(other_browser)
    assert other_browser.get(f"/api/projects/{project['id']}").status_code == 404
    assert client.delete(f"/api/projects/{project['id']}", headers={"X-CSRF-Token": token}).status_code == 204


def test_job_upload_runs_pipeline_and_exposes_scoped_downloads(client):
    token = csrf(client)
    response = client.post("/api/jobs", headers={"X-CSRF-Token": token}, data={
        "audio": (io.BytesIO(b"fake audio"), "song.mp3"),
        "title": "Test song", "style": "scene-hero", "duration": "4", "start": "0",
        "size": "360x640", "fps": "24", "romanize": "false",
    }, content_type="multipart/form-data")
    assert response.status_code == 202
    job = response.get_json()["job"]
    assert job["status"] == "completed"
    assert job["progress"] == 1
    assert client.application.extensions["fake_video_calls"]["auto_select"] is True
    assert client.get(job["download_url"]).data == b"mock video"
    assert client.get(job["preview_url"]).status_code == 200
    assert b"Timing method" in client.get(job["review_url"]).data
    assert b"hello" in client.get(job["lyrics_url"]).data
    assert client.delete(f"/api/jobs/{job['id']}", headers={"X-CSRF-Token": token}).status_code == 204
    assert client.get(job["download_url"]).status_code == 404


def test_job_rejects_unsupported_audio_and_invalid_style(client):
    token = csrf(client)
    bad_file = client.post("/api/jobs", headers={"X-CSRF-Token": token}, data={
        "audio": (io.BytesIO(b"no"), "song.exe"), "style": "scene-hero", "duration": "4"
    }, content_type="multipart/form-data")
    assert bad_file.status_code == 400
    bad_style = client.post("/api/jobs", headers={"X-CSRF-Token": token}, data={
        "audio": (io.BytesIO(b"audio"), "song.mp3"), "style": "untrusted-style", "duration": "4"
    }, content_type="multipart/form-data")
    assert bad_style.status_code == 400

