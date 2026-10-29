"""Experimental Romanized-Hindi forced alignment using parseh/aligner-hi.

This module is opt-in. It returns AsrWord-compatible records with the original
Romanized tokens and timestamps obtained by aligning mapped Devanagari text.
The model is experimental for sung audio; validate the result before relying on it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

# Practical mapping for tokens used by the supplied Are Re Are lyric sheet.
# "maine" is correctly represented as "मैंने" (not "मैने").
_ROMAN_TO_DEVA = {
    "aa": "\u0906", "aag": "\u0906\u0917", "aage": "\u0906\u0917\u0947",
    "aaj": "\u0906\u091c", "aaise": "\u0910\u0938\u093e",
    "afsana": "\u0905\u092b\u0938\u093e\u0928\u093e", "are": "\u0905\u0930\u0947",
    "aur": "\u0914\u0930", "baat": "\u092c\u093e\u0924", "baithe": "\u092c\u0948\u0920\u0947",
    "ban": "\u092c\u0928", "banta": "\u092c\u0928\u0924\u093e", "chale": "\u091a\u0932\u0947",
    "chhu": "\u091b\u0942", "de": "\u0926\u0947", "dekh": "\u0926\u0947\u0916",
    "dhadkane": "\u0927\u0921\u093c\u0915\u0928\u0947", "dil": "\u0926\u093f\u0932",
    "gaya": "\u0917\u092f\u093e", "gayi": "\u0917\u0908", "ha": "\u0939\u093e",
    "haath": "\u0939\u093e\u0925", "hai": "\u0939\u0948", "ho": "\u0939\u094b",
    "hoti": "\u0939\u094b\u0924\u0940", "hua": "\u0939\u0941\u0906", "hum": "\u0939\u092e",
    "humko": "\u0939\u092e\u0915\u094b", "jaana": "\u091c\u093e\u0928\u093e",
    "jaaye": "\u091c\u093e\u090f", "jab": "\u091c\u092c", "jahan": "\u091c\u0939\u093e\u0901",
    "jaise": "\u091c\u0948\u0938\u0947", "jao": "\u091c\u093e\u0913", "kahe": "\u0915\u0939\u0947",
    "kahin": "\u0915\u0939\u0940\u0902", "ke": "\u0915\u0947", "ki": "\u0915\u0940",
    "kisi": "\u0915\u093f\u0938\u0940", "ko": "\u0915\u094b", "koi": "\u0915\u094b\u0908",
    "kuch": "\u0915\u0941\u091b", "kya": "\u0915\u094d\u092f\u093e", "la": "\u0932\u093e",
    "lag": "\u0932\u0917", "lagta": "\u0932\u0917\u0924\u093e", "liya": "\u0932\u093f\u092f\u093e",
    "lo": "\u0932\u094b", "machal": "\u092e\u091a\u0932", "maine": "\u092e\u0948\u0902\u0928\u0947",
    "mausam": "\u092e\u094c\u0938\u092e", "mein": "\u092e\u0947\u0902", "mera": "\u092e\u0947\u0930\u093e",
    "na": "\u0928\u093e", "naam": "\u0928\u093e\u092e", "nahin": "\u0928\u0939\u0940\u0902",
    "ne": "\u0928\u0947", "nikal": "\u0928\u093f\u0915\u0932", "pehchana": "\u092a\u0939\u091a\u093e\u0928\u093e",
    "raat": "\u0930\u093e\u0924", "rahe": "\u0930\u0939\u0947", "raho": "\u0930\u0939\u094b",
    "re": "\u0930\u0947", "ru": "\u0930\u0942", "saamne": "\u0938\u093e\u092e\u0928\u0947",
    "saath": "\u0938\u093e\u0925", "sambhal": "\u0938\u0902\u092d\u0932", "se": "\u0938\u0947",
    "shabnam": "\u0936\u092c\u0928\u092e", "tak": "\u0924\u0915", "tha": "\u0925\u093e",
    "thaam": "\u0925\u093e\u092e", "toh": "\u0924\u094b", "tu": "\u0924\u0942",
    "tum": "\u0924\u0941\u092e", "wahin": "\u0935\u0939\u0940\u0902", "waqt": "\u0935\u0915\u094d\u0924",
    "yakeen": "\u092f\u0915\u0940\u0928", "yeh": "\u092f\u0947", "zara": "\u091c\u093c\u0930\u093e",
}

_TOKEN_CLEAN = re.compile(r"[^A-Za-z]+")


def _get_source_tokens(lines: Iterable) -> list[tuple[str, str]]:
    """Return (display token, Devanagari token) pairs, omitting translations and punctuation."""
    from .align import _looks_like_english_translation, _split_words

    result: list[tuple[str, str]] = []
    unmapped: set[str] = set()
    for line in lines:
        line_text = str(getattr(line, "text", ""))
        if _looks_like_english_translation(line_text):
            continue
        for raw in _split_words(line_text):
            token = str(raw).strip(".,!?;:()[]{}\"'“”‘’").lower()
            token = _TOKEN_CLEAN.sub("", token)
            if not token:
                continue
            deva = _ROMAN_TO_DEVA.get(token)
            if not deva:
                unmapped.add(token)
                continue
            result.append((str(raw), deva))
    if unmapped:
        raise ValueError(
            "Hindi CTC transliteration has unmapped lyric tokens: "
            + ", ".join(sorted(unmapped))
        )
    if not result:
        raise ValueError("No Hindi sung lyric tokens were found for forced alignment.")
    return result


def _prepare_audio(audio_path: str | Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH; it is required for Hindi CTC alignment.") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or "").strip()
        raise RuntimeError(f"Could not convert audio for Hindi CTC alignment: {details}") from exc


def _viterbi_path(log_probs: np.ndarray, labels: list[int], blank: int) -> np.ndarray:
    """Vectorized CTC Viterbi path for a known label sequence."""
    T, _ = log_probs.shape
    if not labels:
        raise ValueError("No labels supplied to CTC aligner.")
    if T < len(labels):
        raise ValueError(f"Audio too short for lyric text ({T} frames for {len(labels)} labels).")

    S = 2 * len(labels) + 1
    extended = np.full(S, blank, dtype=np.int32)
    extended[1::2] = labels
    neg = np.float32(-1e30)

    prev = np.full(S, neg, dtype=np.float32)
    prev[0] = log_probs[0, blank]
    if S > 1:
        prev[1] = log_probs[0, extended[1]]

    # One byte per frame/state for back pointers; much smaller than float DP storage.
    back = np.zeros((T, S), dtype=np.uint8)
    skip_allowed = np.zeros(S, dtype=bool)
    if S > 2:
        skip_allowed[2:] = (extended[2:] != blank) & (extended[2:] != extended[:-2])

    arange = np.arange(S)
    for t in range(1, T):
        best = prev.copy()  # stay in the same CTC state
        move = np.zeros(S, dtype=np.uint8)

        step1 = np.full(S, neg, dtype=np.float32)
        step1[1:] = prev[:-1]
        take1 = step1 > best
        best[take1] = step1[take1]
        move[take1] = 1

        step2 = np.full(S, neg, dtype=np.float32)
        step2[2:] = prev[:-2]
        take2 = (step2 > best) & skip_allowed
        best[take2] = step2[take2]
        move[take2] = 2

        prev = best + log_probs[t, extended]
        back[t] = move

    if S == 1:
        state = 0
    else:
        state = S - 1 if prev[S - 1] >= prev[S - 2] else S - 2
    if prev[state] <= neg / 2:
        raise RuntimeError("No valid CTC alignment path exists for this audio and lyric sequence.")

    path = np.empty(T, dtype=np.int32)
    path[-1] = state
    for t in range(T - 1, 0, -1):
        state -= int(back[t, state])
        path[t - 1] = state
    return path


def align_romanized_hindi(lines: Iterable, audio_path: str | Path):
    """Return AsrWord-compatible records timed by the Hindi CTC model.

    This is experimental for sung lyrics. Output retains the user's original
    Romanized tokens so the existing aligner can consume it without changing
    displayed text. English translation rows are excluded from the CTC text.
    """
    import os
    from .asr import AsrWord

    source_tokens = _get_source_tokens(lines)

    # Cache model files inside this project if the caller has not set a cache.
    cache_home = Path(__file__).resolve().parents[1] / ".hf_cache"
    os.environ.setdefault("HF_HOME", str(cache_home))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])

    from huggingface_hub import hf_hub_download
    import onnxruntime as ort

    repo = "parseh/aligner-hi"
    meta_path = hf_hub_download(repo, "meta.json")
    vocab_path = hf_hub_download(repo, "vocab.json")
    with open(meta_path, encoding="utf-8") as handle:
        config = json.load(handle)["model"]
    with open(vocab_path, encoding="utf-8") as handle:
        vocab = json.load(handle)

    # Keep each token's character interval in the contiguous CTC label sequence.
    chars: list[str] = []
    spans: list[tuple[str, int, int]] = []
    for i, (display_token, deva_token) in enumerate(source_tokens):
        start = len(chars)
        chars.extend(deva_token)
        end = len(chars)
        spans.append((display_token, start, end))
        if i < len(source_tokens) - 1:
            chars.append(" ")

    delimiter = config["word_delimiter"]
    unknown = sorted({ch for ch in chars if ch != " " and ch not in vocab})
    if unknown:
        raise ValueError("Hindi CTC model vocabulary lacks characters: " + repr(unknown))
    labels = [int(vocab[delimiter if ch == " " else ch]) for ch in chars]

    with tempfile.TemporaryDirectory(prefix="hero_hindi_ctc_") as temp_dir:
        wav_path = Path(temp_dir) / "audio_16k_mono.wav"
        _prepare_audio(audio_path, wav_path)
        with wave.open(str(wav_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

        if sample_rate != int(config["sample_rate"]) or channels != 1:
            raise RuntimeError(
                f"Expected mono {config['sample_rate']} Hz audio; got {channels} channel(s), {sample_rate} Hz."
            )
        if not len(audio):
            raise RuntimeError("The converted audio is empty.")
        audio -= float(audio.mean())
        audio /= max(float(audio.std()), 1e-7)

        model_path = hf_hub_download(repo, config["file"])
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        logits = session.run(
            [config["output"]],
            {config["input"]: audio[np.newaxis, :].astype(np.float32)},
        )[0]
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 3:
            logits = logits[0]
        if logits.ndim != 2 or logits.shape[1] != int(config["vocab_size"]):
            raise RuntimeError(f"Unexpected Hindi CTC output shape: {logits.shape}")

        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        path = _viterbi_path(log_probs, labels, int(config["blank_id"]))

        # Convert odd CTC states back to character indices and gather their frame bounds.
        active = (path % 2) == 1
        active_frames = np.flatnonzero(active)
        char_positions = ((path[active] - 1) // 2).astype(np.int32)
        char_starts = np.full(len(labels), len(path), dtype=np.int32)
        char_ends = np.full(len(labels), -1, dtype=np.int32)
        np.minimum.at(char_starts, char_positions, active_frames)
        np.maximum.at(char_ends, char_positions, active_frames)

        frame_seconds = float(config["frame_samples"]) / float(config["sample_rate"])
        result = []
        for token, start_char, end_char in spans:
            starts = char_starts[start_char:end_char]
            ends = char_ends[start_char:end_char]
            valid = ends >= starts
            valid_starts = starts[valid]
            valid_ends = ends[valid]
            if not len(valid_starts):
                continue
            start = float(valid_starts.min()) * frame_seconds
            end = float(valid_ends.max() + 1) * frame_seconds
            if end <= start:
                continue
            # Keep all forced tokens available to the existing matcher. Their confidence
            # is not represented as ASR confidence; independent review is required.
            result.append(AsrWord(text=token, start=start, end=end, prob=0.5))

    if not result:
        raise RuntimeError("Hindi CTC aligner produced no usable word spans.")
    return result


def main() -> int:
    """Diagnostic CLI: align the supplied lyrics file against the full audio."""
    import argparse
    from .lyrics import parse_lyrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default="test_input/song.mp3")
    parser.add_argument("--lyrics", default="test_input/lyrics.txt")
    args = parser.parse_args()
    lyric_path = Path(args.lyrics)
    lines, _timed = parse_lyrics(lyric_path.read_text(encoding="utf-8-sig"))
    words = align_romanized_hindi(lines, args.audio)
    print(f"CTC forced-alignment output: {len(words)} word spans")
    print("First 15 spans:")
    for word in words[:15]:
        print(f"{word.start:7.2f}-{word.end:7.2f}  {word.text}")
    print("\nSpans around 286-310 seconds:")
    selected = [w for w in words if 286 <= w.start <= 310]
    for word in selected:
        print(f"{word.start:7.2f}-{word.end:7.2f}  {word.text}")
    print("\nLast 15 spans:")
    for word in words[-15:]:
        print(f"{word.start:7.2f}-{word.end:7.2f}  {word.text}")
    print("\nNOTE: This result is experimental; do not integrate it until timestamps are reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
