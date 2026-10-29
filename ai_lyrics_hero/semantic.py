"""Optional semantic layer: lyric line -> {scene, mood, confidence}.

* LocalSemantic (always available, deterministic): exact dictionary words + a small mood word list.
* LLMSemantic (optional): a configured OpenAI-compatible chat endpoint. NEVER required: with no key, no
  network, a timeout, an HTTP error or an invalid answer, the caller silently (logged) uses LocalSemantic.

Safety rules: only the lyric text is sent; the model can only choose from whitelisted scene/mood names;
anything else is discarded; the returned data is never used as display text, so lyrics cannot be altered."""
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional

from .scenes import SceneLexicon, SCENE_KINDS
from .text_utils import normalize

log = logging.getLogger("ai_lyrics_hero")
MOODS = ("romantic", "sad", "energetic", "calm", "dreamy", "intense", "joyful", "neutral")
MIN_CONFIDENCE = 0.6

_MOOD_WORDS = {
    "romantic": "love pyar pyaar ishq mohabbat kiss dil heart sanam saajan jaan baby darling chumma".split(),
    "sad": "tears aansu rona roye judaai bichad alone akela tanha dard pain cry broken toota yaad".split(),
    "energetic": "high fire aag dance naach party run jump loud wild rock shake".split(),
    "calm": "sukoon shaanti peace quiet slow dheere soft chaand moon night raat".split(),
    "dreamy": "dream sapna sapne khwab neend sleep yaadein memories stars taare".split(),
    "intense": "jalta burning aag fire toofan storm war ladai jung".split(),
    "joyful": "khushi happy smile muskaan hasna laugh celebrate".split(),
}
_MOOD_WORDS = {k: {normalize(w) for w in v} for k, v in _MOOD_WORDS.items()}


@dataclass
class Semantic:
    scene: Optional[str] = None
    mood: str = "neutral"
    confidence: float = 0.0
    source: str = "local"


class LocalSemantic:
    name = "local"

    def __init__(self, lexicon=None, allow_ambiguous=False):
        self.lex = lexicon or SceneLexicon()
        self.allow_ambiguous = allow_ambiguous

    def classify(self, lines: List[List[str]]) -> Dict[int, Semantic]:
        out = {}
        for i, toks in enumerate(lines):
            kind, _, _ = self.lex.match(toks, self.allow_ambiguous)
            norm = [normalize(t) for t in toks]
            scores = {m: sum(1 for w in norm if w in ws) for m, ws in _MOOD_WORDS.items()}
            mood, hits = max(scores.items(), key=lambda kv: (kv[1], kv[0]))
            out[i] = Semantic(kind, mood if hits else "neutral", 1.0 if kind else (0.5 if hits else 0.0), "local")
        return out


class LLMSemantic:
    """OpenAI-compatible chat-completions client (stdlib only). Configure via environment variables:
    AI_LYRICS_LLM_API_KEY, AI_LYRICS_LLM_MODEL (default gpt-4o-mini), AI_LYRICS_LLM_URL (default OpenAI)."""
    name = "llm"

    def __init__(self, api_key=None, model=None, url=None, timeout=20.0, opener=None):
        self.api_key = api_key if api_key is not None else os.environ.get("AI_LYRICS_LLM_API_KEY", "")
        self.model = model or os.environ.get("AI_LYRICS_LLM_MODEL", "gpt-4o-mini")
        self.url = url or os.environ.get("AI_LYRICS_LLM_URL", "https://api.openai.com/v1/chat/completions")
        self.timeout = timeout
        self._open = opener or urllib.request.urlopen

    @property
    def configured(self):
        return bool(self.api_key)

    def _prompt(self, lines):
        numbered = "\n".join(f"{i}: {' '.join(t)}" for i, t in enumerate(lines))
        return ("You pick background scenes for a lyric video. For each numbered lyric line answer with a JSON list "
                "of objects {\"i\": int, \"scene\": one of " + json.dumps(SCENE_KINDS) + ", \"mood\": one of " +
                json.dumps(list(MOODS)) + ", \"confidence\": number 0..1}. Use scene \"sky\" when nothing fits. "
                "Answer with JSON only, never rewrite or translate the lyrics.\n" + numbered)

    def classify(self, lines: List[List[str]]) -> Dict[int, Semantic]:
        if not self.configured or not lines:
            return {}
        body = json.dumps({"model": self.model, "temperature": 0,
                           "messages": [{"role": "user", "content": self._prompt(lines)}]}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with self._open(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            return self._validate(text, len(lines))
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError, TypeError) as e:
            log.warning("LLM scene analysis unavailable (%s); using the local analysis", type(e).__name__)
            return {}

    @staticmethod
    def _validate(text, n):
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            items = json.loads(t)
        except ValueError:
            log.warning("LLM answer was not valid JSON; ignored")
            return {}
        out = {}
        if not isinstance(items, list):
            return {}
        for it in items:
            if not isinstance(it, dict):
                continue
            i, scene, mood, conf = it.get("i"), it.get("scene"), it.get("mood", "neutral"), it.get("confidence")
            if not (isinstance(i, int) and 0 <= i < n) or scene not in SCENE_KINDS:
                continue
            if mood not in MOODS:
                mood = "neutral"
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            if not (0.0 <= conf <= 1.0) or conf < MIN_CONFIDENCE:
                continue
            out[i] = Semantic(scene, mood, conf, "llm")
        return out


def analyze_lines(lines, cfg, lexicon=None, llm=None):
    """Merge local (authoritative for exact words) with optional LLM answers (only fills unmatched lines)."""
    local = LocalSemantic(lexicon, cfg.allow_ambiguous).classify(lines)
    provider = llm if llm is not None else (LLMSemantic() if getattr(cfg, "llm_provider", "none") == "openai" else None)
    if provider is None:
        return local, "local"
    remote = provider.classify(lines)
    if not remote:
        return local, "local (LLM unavailable or gave no valid answer)"
    for i, sem in remote.items():
        if local[i].scene is None or local[i].scene == "sky":
            local[i] = Semantic(sem.scene if sem.scene != "sky" else None, sem.mood, sem.confidence, "llm")
        elif local[i].mood == "neutral":
            local[i].mood = sem.mood
    return local, "local + LLM"
