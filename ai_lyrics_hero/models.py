from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class LyricEvent:
    start: float
    end: float
    words: List[Word]
    explicit_hero: Optional[Tuple[int, int]] = None   # inclusive word index range
    timing_reliable: bool = True
    hero: Optional[Tuple[int, int]] = None            # filled by hero selection
    accent: Optional[str] = None                      # palette name chosen by the visual director

    @property
    def duration(self):
        return self.end - self.start

    @property
    def text(self):
        return " ".join(w.text for w in self.words)
