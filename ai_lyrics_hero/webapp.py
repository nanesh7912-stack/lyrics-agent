"""Browser interface and same-origin API for AI Lyrics Hero."""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, g, jsonify, request, send_file, send_from_directory, session
from werkzeug.utils import secure_filename

from .config import HeroConfig, STYLE_PRESETS


ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mp4"}
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
PROJECT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
DEFAULT_CONTACT = "nanesh7912@gmail.com"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    if len(json.dumps(value, ensure_ascii=False)) > 128_000:
        raise ValueError(f"{label} is too large")
    return value


def _connect(path):
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def _transaction(path):
    conn = _connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_db(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _transaction(path) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                title TEXT NOT NULL,
                settings TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS project_owner_updated ON projects(owner, updated_at DESC);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                settings TEXT NOT NULL,
                work_dir TEXT NOT NULL,
                output_path TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS job_owner_updated ON jobs(owner, updated_at DESC);
            """
        )


def _session_secret(data_dir):
    configured = os.environ.get("AI_LYRICS_HERO_SECRET_KEY")
    if configured:
        return configured
    secret_path = Path(data_dir) / ".web_session_secret"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return secret_path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        value = secrets.token_hex(32)
        try:
            with secret_path.open("x", encoding="ascii") as handle:
                handle.write(value)
            try:
                secret_path.chmod(0o600)
            except OSError:
                pass
            return value
        except FileExistsError:
            return secret_path.read_text(encoding="ascii").strip()


def _default_data_dir():
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return base / "AI Lyrics Hero"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "ai-lyrics-hero"


def create_app(test_config=None):
    app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="/assets")
    overrides = dict(test_config or {})
    configured_data_dir = (overrides.get("DATA_DIR") or os.environ.get("AI_LYRICS_HERO_DATA_DIR")
                           or (Path(overrides["DATABASE"]).parent if overrides.get("DATABASE") else _default_data_dir()))
    data_dir = Path(configured_data_dir).expanduser().resolve()
    app.config.from_mapping(
        SECRET_KEY=overrides.get("SECRET_KEY") or _session_secret(data_dir),
        DATABASE=overrides.get("DATABASE", str(data_dir / "lyrics_hero.sqlite3")),
        DATA_DIR=str(data_dir),
        MAX_CONTENT_LENGTH=105 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("AI_LYRICS_HERO_COOKIE_SECURE", "0") == "1",
        PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30,
    )
    if overrides:
        app.config.update(overrides)
    _init_db(app.config["DATABASE"])
    with _transaction(app.config["DATABASE"]) as startup_db:
        startup_db.execute("UPDATE jobs SET status='failed', stage='Server restarted; please create the video again', "
                           "error='The server restarted before this render finished.', updated_at=? "
                           "WHERE status IN ('queued','running')", (_now(),))
    Path(app.config["DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    app.extensions["lyrics_executor"] = app.config.get("JOB_EXECUTOR") or ThreadPoolExecutor(max_workers=1)
    app.extensions["lyrics_executor_owned"] = "JOB_EXECUTOR" not in (test_config or {})
    lock = threading.Lock()

    def db():
        if "db" not in g:
            g.db = _connect(app.config["DATABASE"])
        return g.db

    @app.teardown_appcontext
    def close_db(_error=None):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    @app.before_request
    def require_csrf_for_writes():
        if request.path.startswith("/api/") and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            expected = session.get("csrf_token", "")
            supplied = request.headers.get("X-CSRF-Token", "")
            if not expected or not supplied or not hmac.compare_digest(expected, supplied):
                return jsonify(error="Your session expired. Refresh the page and try again."), 403
        return None

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Content-Security-Policy",
            # Vue's global build compiles the fixed, bundled template at mount time.
            # User-provided text only enters escaped text nodes; it is never compiled as a template.
            "default-src 'self'; script-src 'self' https://unpkg.com 'unsafe-eval'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
            "font-src 'self' data:; base-uri 'self'; form-action 'self' mailto:; frame-ancestors 'none'",
        )
        return response

    def owner_id():
        ident = session.get("browser_id")
        if not ident:
            ident = uuid.uuid4().hex
            session["browser_id"] = ident
            session.permanent = True
        return ident

    def csrf_token():
        token = session.get("csrf_token")
        if not token:
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return token

    def row_to_project(row):
        return {"id": row["id"], "title": row["title"], "settings": json.loads(row["settings"]),
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def row_to_job(row):
        item = {"id": row["id"], "title": row["title"], "status": row["status"], "stage": row["stage"],
                "progress": row["progress"], "error": row["error"], "created_at": row["created_at"],
                "updated_at": row["updated_at"]}
        if row["status"] == "completed":
            item.update(download_url=f"/api/jobs/{row['id']}/download",
                        preview_url=f"/api/jobs/{row['id']}/download?inline=1",
                        review_url=f"/api/jobs/{row['id']}/review",
                        lyrics_url=f"/api/jobs/{row['id']}/lyrics")
        return item

    def owned_project(project_id):
        if not PROJECT_ID_RE.fullmatch(project_id):
            abort(404)
        row = db().execute("SELECT * FROM projects WHERE id=? AND owner=?", (project_id, owner_id())).fetchone()
        if row is None:
            abort(404)
        return row

    def owned_job(job_id):
        if not JOB_ID_RE.fullmatch(job_id):
            abort(404)
        row = db().execute("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner_id())).fetchone()
        if row is None:
            abort(404)
        return row

    def update_job(job_id, **fields):
        fields["updated_at"] = _now()
        allowed = {"status", "stage", "progress", "error", "updated_at"}
        fields = {key: value for key, value in fields.items() if key in allowed}
        sql = ", ".join(f"{key}=?" for key in fields)
        with _transaction(app.config["DATABASE"]) as conn:
            conn.execute(f"UPDATE jobs SET {sql} WHERE id=?", (*fields.values(), job_id))

    def run_job(job_id, audio_path, lyrics_text, settings):
        with app.app_context():
            stage = ("Analyzing the full song and selecting its best lyric section"
                     if settings.get("auto_select") else "Preparing audio and lyrics")
            update_job(job_id, status="running", stage=stage, progress=0.01)
            try:
                from .workflow import make_video

                style = settings["style"]
                config_data = dict(STYLE_PRESETS[style])
                config_data.update(title=settings["title"], target_duration=settings["duration"])
                config = HeroConfig.from_dict(config_data)
                output_path = Path(settings["output_path"])

                def progress(value):
                    update_job(job_id, status="running", stage="Rendering video",
                               progress=max(0.02, min(0.98, float(value))))

                report = make_video(
                    audio_path, str(output_path), lyrics_text or None, config,
                    size=(settings["width"], settings["height"]), fps=settings["fps"],
                    start=settings["start"], end=None, encoder="auto", language=settings["language"] or None,
                    use_asr=True, model_size=settings["model"], progress=progress,
                    romanize=settings["romanize"], auto_select=settings["auto_select"],
                    track_title=settings["track_title"], artist=settings["artist"],
                    lyrics_lookup=settings["lyrics_lookup"],
                )
                # Store the report without machine-specific absolute paths.
                auto_lyrics = report.get("auto_lyrics")
                if auto_lyrics:
                    report["auto_lyrics"] = Path(auto_lyrics).name
                with _transaction(app.config["DATABASE"]) as conn:
                    conn.execute("UPDATE jobs SET status='completed', stage='Video ready', progress=1, error='', updated_at=? WHERE id=?",
                                 (_now(), job_id))
                return report
            except Exception as exc:
                update_job(job_id, status="failed", stage="Could not finish the video", error=str(exc)[:1200])

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/<path:page>")
    def spa_page(page):
        if page in {"about", "services", "contact", "help", "projects", "templates", "settings"}:
            return send_from_directory(WEB_DIR, "index.html")
        abort(404)

    @app.get("/api/session")
    def session_info():
        owner_id()
        return jsonify(csrf_token=csrf_token(), contact_email=DEFAULT_CONTACT)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", service="AI Lyrics Hero")

    @app.route("/api/projects", methods=["GET", "POST"])
    def projects_collection():
        if request.method == "GET":
            rows = db().execute("SELECT * FROM projects WHERE owner=? ORDER BY updated_at DESC", (owner_id(),)).fetchall()
            return jsonify(projects=[row_to_project(row) for row in rows])
        payload = request.get_json(silent=True) or {}
        try:
            title = str(payload.get("title", "Untitled project")).strip()[:100] or "Untitled project"
            settings = _json_object(payload.get("settings", {}), "settings")
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        ident, now = uuid.uuid4().hex, _now()
        db().execute("INSERT INTO projects(id,owner,title,settings,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                     (ident, owner_id(), title, json.dumps(settings, ensure_ascii=False), now, now))
        db().commit()
        return jsonify(project={"id": ident, "title": title, "settings": settings,
                                "created_at": now, "updated_at": now}), 201

    @app.route("/api/projects/<project_id>", methods=["GET", "PUT", "DELETE"])
    def project_item(project_id):
        row = owned_project(project_id)
        if request.method == "GET":
            return jsonify(project=row_to_project(row))
        if request.method == "DELETE":
            db().execute("DELETE FROM projects WHERE id=? AND owner=?", (project_id, owner_id()))
            db().commit()
            return "", 204
        payload = request.get_json(silent=True) or {}
        try:
            title = str(payload.get("title", row["title"])).strip()[:100] or "Untitled project"
            settings = _json_object(payload.get("settings", json.loads(row["settings"])), "settings")
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        now = _now()
        db().execute("UPDATE projects SET title=?, settings=?, updated_at=? WHERE id=? AND owner=?",
                     (title, json.dumps(settings, ensure_ascii=False), now, project_id, owner_id()))
        db().commit()
        return jsonify(project={"id": project_id, "title": title, "settings": settings,
                                "created_at": row["created_at"], "updated_at": now})

    @app.get("/api/jobs")
    def jobs_collection():
        rows = db().execute("SELECT * FROM jobs WHERE owner=? ORDER BY updated_at DESC LIMIT 50", (owner_id(),)).fetchall()
        return jsonify(jobs=[row_to_job(row) for row in rows])

    @app.post("/api/jobs")
    def create_job():
        owner = owner_id()
        audio = request.files.get("audio")
        if audio is None or not audio.filename:
            return jsonify(error="Choose an audio file first."), 400
        safe_name = secure_filename(audio.filename)
        if Path(safe_name).suffix.lower() not in ALLOWED_AUDIO:
            return jsonify(error="Use an MP3, WAV, M4A, AAC, FLAC, OGG, OPUS, WMA, or MP4 audio file."), 400
        active = db().execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
        if active >= 2:
            return jsonify(error="The video queue is full. Wait for a video to finish, then try again."), 429
        payload = request.form
        try:
            style = payload.get("style", "scene-hero")
            if style not in STYLE_PRESETS:
                raise ValueError("Choose one of the available video styles.")
            language = payload.get("language", "").strip().lower() or None
            if language and language not in {"en", "hi", "es", "fr", "pt", "ja"}:
                raise ValueError("Choose one of the listed transcription languages or use auto-detect.")
            start = float(payload.get("start", "0"))
            duration = float(payload.get("duration", "40"))
            fps = int(payload.get("fps", "30"))
            width, height = (int(part) for part in payload.get("size", "1080x1920").lower().split("x", 1))
            if not 0 <= start < 24 * 60 * 60:
                raise ValueError("Start time must be between 0 and 24 hours.")
            if not 2 <= duration <= 600:
                raise ValueError("Video length must be between 2 and 600 seconds.")
            if fps not in {24, 25, 30, 60}:
                raise ValueError("Choose 24, 25, 30, or 60 frames per second.")
            if width < 360 or height < 360 or width > 1920 or height > 1920:
                raise ValueError("Video dimensions must be between 360 and 1920 pixels.")
        except (ValueError, TypeError):
            return jsonify(error="Check the video length, start time, size, and frame rate."), 400
        lyrics_text = (payload.get("lyrics_text") or "").strip()
        if len(lyrics_text) > 100_000:
            return jsonify(error="Lyrics are too long. The limit is 100,000 characters."), 400
        ident, now = uuid.uuid4().hex, _now()
        job_dir = Path(app.config["DATA_DIR"]) / "jobs" / owner / ident
        job_dir.mkdir(parents=True, exist_ok=True)
        audio_path = job_dir / ("source" + Path(safe_name).suffix.lower())
        output_path = job_dir / "lyric-video.mp4"
        try:
            audio.save(audio_path)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify(error="The audio upload could not be saved. Try a smaller file."), 400
        if audio_path.stat().st_size > 100 * 1024 * 1024:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify(error="Audio files must be smaller than 100 MB."), 413
        settings = {
            "title": str(payload.get("title", "My lyric video")).strip()[:100] or "My lyric video",
            "style": style, "language": language or "",
            "romanize": payload.get("romanize", "false").lower() in {"true", "1", "on"},
            "auto_select": payload.get("auto_select", "true").lower() in {"true", "1", "on"},
            "start": start, "duration": duration, "fps": fps, "width": width, "height": height,
            "model": payload.get("model", "medium") if payload.get("model") in {"tiny", "base", "small", "medium"} else "medium",
            "track_title": str(payload.get("track_title", "")).strip()[:200],
            "artist": str(payload.get("artist", "")).strip()[:200],
            "lyrics_lookup": payload.get("lyrics_lookup", "true").lower() in {"true", "1", "on"},
            "output_path": str(output_path),
        }
        db().execute("INSERT INTO jobs(id,owner,title,status,stage,progress,settings,work_dir,output_path,error,created_at,updated_at) "
                     "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                     (ident, owner, settings["title"], "queued", "Waiting to start", 0,
                      json.dumps({k: v for k, v in settings.items() if k != "output_path"}), str(job_dir),
                      str(output_path), "", now, now))
        db().commit()
        with lock:
            app.extensions["lyrics_executor"].submit(run_job, ident, str(audio_path), lyrics_text, settings)
        row = db().execute("SELECT * FROM jobs WHERE id=?", (ident,)).fetchone()
        return jsonify(job=row_to_job(row)), 202

    @app.get("/api/jobs/<job_id>")
    def job_item(job_id):
        return jsonify(job=row_to_job(owned_job(job_id)))

    @app.delete("/api/jobs/<job_id>")
    def delete_job(job_id):
        row = owned_job(job_id)
        if row["status"] in {"queued", "running"}:
            return jsonify(error="A running video cannot be deleted yet."), 409
        db().execute("DELETE FROM jobs WHERE id=? AND owner=?", (job_id, owner_id()))
        db().commit()
        shutil.rmtree(row["work_dir"], ignore_errors=True)
        return "", 204

    @app.get("/api/jobs/<job_id>/download")
    def download_job(job_id):
        row = owned_job(job_id)
        if row["status"] != "completed" or not Path(row["output_path"]).is_file():
            abort(404)
        return send_file(row["output_path"], mimetype="video/mp4", as_attachment=request.args.get("inline") != "1",
                         download_name="lyrics-video.mp4", conditional=True, max_age=0)

    @app.get("/api/jobs/<job_id>/review")
    def job_review(job_id):
        row = owned_job(job_id)
        if row["status"] != "completed":
            abort(404)
        path = Path(row["output_path"]).with_suffix(".review.txt")
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="text/plain; charset=utf-8", as_attachment=True,
                         download_name="lyrics-video-review.txt", max_age=0)

    @app.get("/api/jobs/<job_id>/lyrics")
    def job_lyrics(job_id):
        row = owned_job(job_id)
        if row["status"] != "completed":
            abort(404)
        path = Path(row["output_path"]).with_suffix(".auto.lrc")
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="text/plain; charset=utf-8", as_attachment=True,
                         download_name="lyrics-timed.auto.lrc", max_age=0)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="Audio files must be smaller than 100 MB."), 413

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify(error="This item was not found in your browser session."), 404
        return jsonify(error="Page not found."), 404

    return app


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Run the AI Lyrics Hero browser app")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args(argv)
    app = create_app()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
