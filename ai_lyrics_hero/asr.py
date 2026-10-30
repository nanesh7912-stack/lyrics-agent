"""
Faster-Whisper transcription backend for AI Lyrics Hero.

Word-level timestamps are preserved.
VAD is disabled by default because it may filter out sung vocals.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile


@dataclass
class AsrWord:
    """Recognized word with timestamps and confidence."""

    text: str
    start: float
    end: float
    prob: float = 1.0


def _configure_model_cache() -> Path:
    """Choose a writable Hugging Face cache before importing Faster-Whisper.

    A stale HF_HOME can point at a protected Windows profile folder. Prefer an
    explicitly configured AI_LYRICS_HERO_MODEL_CACHE, then a writable existing
    Hugging Face cache, and finally the project's local caches.
    """
    project = Path(__file__).resolve().parents[1]
    candidates = []

    explicit = os.environ.get("AI_LYRICS_HERO_MODEL_CACHE")
    if explicit:
        candidates.append((Path(explicit).expanduser(), True))

    configured_home = os.environ.get("HF_HOME")
    if configured_home:
        candidates.append((Path(configured_home).expanduser(), False))

    # These project caches may already contain downloaded Whisper models.
    candidates.extend([
        (project / "model_cache" / "huggingface", False),
        (project / ".hf_cache", False),
        (Path(tempfile.gettempdir()) / "ai_lyrics_hero_hf_cache", False),
    ])

    failures = []
    for home, is_explicit in candidates:
        try:
            home.mkdir(parents=True, exist_ok=True)
            hub_cache = home / "hub"
            hub_cache.mkdir(parents=True, exist_ok=True)
            # Verify actual write access; os.access alone is unreliable on Windows.
            with tempfile.NamedTemporaryFile(prefix=".ai_lyrics_hero_", dir=home, delete=True):
                pass
        except OSError as exc:
            failures.append(f"{home}: {exc}")
            if is_explicit:
                raise RuntimeError(
                    f"The model cache in AI_LYRICS_HERO_MODEL_CACHE is not writable: {home}. "
                    "Choose a writable folder and try again."
                ) from exc
            continue

        # Set these unconditionally so an inaccessible system default cannot
        # override the selected writable cache.
        os.environ["HF_HOME"] = str(home)
        os.environ["HF_HUB_CACHE"] = str(hub_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
        return home

    detail = "; ".join(failures)
    raise RuntimeError(
        "Could not find a writable model cache. Set AI_LYRICS_HERO_MODEL_CACHE to a writable folder. "
        + detail
    )


def available() -> bool:
    """Check whether Faster-Whisper is installed."""
    try:
        _configure_model_cache()
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe(
    audio_path,
    language=None,
    model_size="medium",
    device="auto",
    compute_type="auto",
    vad_filter=False,
):
    """
    Transcribe audio and return AsrWord objects.

    VAD defaults to False to avoid filtering out sung vocals.
    Existing callers can continue using the original arguments.
    """
    if not available():
        raise RuntimeError(
            "faster-whisper is not installed. "
            "Run: python -m pip install faster-whisper"
        )

    # Re-select before importing the model backend, overriding stale or
    # inaccessible system cache settings when necessary.
    _configure_model_cache()

    audio = Path(audio_path)

    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    from faster_whisper import WhisperModel

    configured_compute_type = os.environ.get("AI_LYRICS_HERO_ASR_COMPUTE_TYPE", "").strip()
    selected_compute_type = (
        configured_compute_type
        if compute_type == "auto" and configured_compute_type
        else "default" if compute_type == "auto" else compute_type
    )

    model_options = {}
    configured_cpu_threads = os.environ.get("AI_LYRICS_HERO_ASR_CPU_THREADS", "").strip()
    if configured_cpu_threads:
        try:
            cpu_threads = int(configured_cpu_threads)
        except ValueError as exc:
            raise ValueError("AI_LYRICS_HERO_ASR_CPU_THREADS must be a positive integer.") from exc
        if cpu_threads < 1:
            raise ValueError("AI_LYRICS_HERO_ASR_CPU_THREADS must be a positive integer.")
        model_options["cpu_threads"] = cpu_threads

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=selected_compute_type,
        **model_options,
    )

    segments, _info = model.transcribe(
        str(audio),
        language=language,
        word_timestamps=True,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
        beam_size=5,
        temperature=0.0,
    )

    words = []

    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()

            if not text:
                continue

            start = float(word.start)
            end = float(word.end)
            probability = float(
                getattr(word, "probability", 1.0)
            )

            if start < 0 or end <= start:
                continue

            words.append(
                AsrWord(
                    text=text,
                    start=start,
                    end=end,
                    prob=probability,
                )
            )

    return words
