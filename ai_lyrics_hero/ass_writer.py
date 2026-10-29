"""Write LyricEvents as an ASS file with karaoke word timing.

Consecutive events sharing the same time window are rendered as a bilingual pair:
the first row is the sung lyric and the second row is its translation.
"""
from .ass_parser import parse_ass  # noqa: F401 (round-trip is covered by tests)


def _ts(x):
    x = max(0.0, x)
    cs = int(round(x * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _same_window(a, b, tol=0.03):
    return abs(float(a.start) - float(b.start)) <= tol and abs(float(a.end) - float(b.end)) <= tol


_ENGLISH_WORDS = set("a about after again all am an and are as at be because been before between but by can could did do down each for from get give go had has have he her here him his hold how i if in into is it know let like little look love me more my never no not now of oh on one only or other our out over people right said say see she should so someone something story take tell than that the their them then there they thing think this those through time to too up us very was we what when where which who why will with without would you your happened recognizes talking hands hand knows beautiful".split())


def _looks_english_translation(text):
    import re
    tokens = [re.sub(r"[^a-z']", "", token.lower()) for token in text.split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < 3:
        return False
    hits = sum(token in _ENGLISH_WORDS for token in tokens)
    return hits >= 2 and hits / len(tokens) >= 0.45


def _is_bilingual_pair(source, translation):
    return (not _looks_english_translation(source.text)
            and _looks_english_translation(translation.text)
            and _same_window(source, translation))


def write_ass(events, path, width=1080, height=1920, title="Lyrics"):
    import os

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    out = [
        "[Script Info]", f"Title: {title}", "ScriptType: v4.00+",
        f"PlayResX: {width}", f"PlayResY: {height}", "",
        "[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour",
        "Style: Default,Poppins,60,&H00FFFFFF", "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    for idx, ev in enumerate(events):
        pair_next = idx + 1 < len(events) and _is_bilingual_pair(ev, events[idx + 1])
        pair_prev = idx > 0 and _is_bilingual_pair(events[idx - 1], ev)

        # Override positions only for a pair. All other subtitle placement
        # remains controlled by the existing renderer/style.
        prefix = ""
        layer = 0
        if pair_next and not pair_prev:
            prefix = r"{\an2\pos(" + str(width // 2) + "," + str(int(height * 0.755)) + r")\fs60}"
            layer = 0
        elif pair_prev:
            prefix = r"{\an2\pos(" + str(width // 2) + "," + str(int(height * 0.835)) + r")\fs38}"
            layer = 1

        parts = []
        for i, w in enumerate(ev.words):
            nxt = ev.words[i + 1].start if i + 1 < len(ev.words) else ev.end
            cs = max(1, int(round((nxt - w.start) * 100)))
            # Word joiner keeps a literal backslash from becoming an ASS tag
            # such as \N; escaped braces stay visible instead of starting a tag.
            txt = w.text.replace("\\", "\\\u2060").replace("{", r"\{").replace("}", r"\}")
            if ev.explicit_hero and ev.explicit_hero[0] == i:
                txt = "[[" + txt
            if ev.explicit_hero and ev.explicit_hero[1] == i:
                txt += "]]"
            word_prefix = prefix if i == 0 else ""
            parts.append(word_prefix + ("{\\k%d}%s" % (cs, txt)))

        out.append(
            f"Dialogue: {layer},{_ts(ev.start)},{_ts(ev.end)},Default,,0,0,0,," +
            " ".join(parts)
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return path
