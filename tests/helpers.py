from ai_lyrics_hero.models import LyricEvent, Word
from ai_lyrics_hero.ass_parser import interpolate_words


def ev(text, start=0.0, end=3.0, reliable=True, explicit=None):
    return LyricEvent(start, end, interpolate_words(text.split(), start, end), explicit, reliable)
