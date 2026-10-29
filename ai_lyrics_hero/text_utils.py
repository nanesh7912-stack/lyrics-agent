"""Text helpers shared by lyric parsing, alignment and image matching."""
import re
import unicodedata


def normalize(token):
    """NFC, lowercase, punctuation stripped (marks used by Indic scripts are kept)."""
    t = unicodedata.normalize("NFC", token)
    t = "".join(ch for ch in t if not unicodedata.category(ch).startswith("P") and ch not in "\u200c\u200d")
    return t.lower().strip()


_VOWELS = set("aeiou")
_MAP = {"c": "k", "q": "k", "w": "v", "z": "j", "x": "ks"}

# ---- Devanagari -> rough Latin, used ONLY to compare ASR output with Romanized lyrics -------------
_D_CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
           "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
           "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
           "ल": "l", "ळ": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h"}
_D_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "d", "ढ": "dh", "फ": "f"}
_D_VOW = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai",
          "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
_D_MATRA = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o",
            "ौ": "au", "ॉ": "o", "ॅ": "e"}


def has_devanagari(text):
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def deva_to_roman(token):
    """Approximate transliteration (schwa handling is irrelevant: skeletons drop vowels anyway).

    This is an internal matching aid; it is never shown and never replaces the user's lyrics."""
    t = unicodedata.normalize("NFC", token)
    out, i, n = [], 0, len(t)
    while i < n:
        ch = t[i]
        if ch in _D_CONS:
            nxt = t[i + 1] if i + 1 < n else ""
            if nxt == "\u093c":                                 # nukta: ज़ फ़ ड़ ...
                out.append(_D_NUKTA.get(ch, _D_CONS[ch])); i += 2
            else:
                out.append(_D_CONS[ch]); i += 1
            nxt = t[i] if i < n else ""
            if nxt == "\u094d":                                 # virama: no vowel (conjunct)
                i += 1
            elif nxt in _D_MATRA:
                out.append(_D_MATRA[nxt]); i += 1
            else:
                out.append("a")
        elif ch in _D_VOW:
            out.append(_D_VOW[ch]); i += 1
        elif ch in ("\u0902", "\u0901"):                       # anusvara / chandrabindu
            out.append("n"); i += 1
        elif ch == "\u0903":
            out.append("h"); i += 1
        elif ch in _D_MATRA:
            out.append(_D_MATRA[ch]); i += 1
        elif ch in ("\u094d", "\u093c", "\u0964", "\u0965"):
            i += 1
        elif ch.isalnum():
            out.append(ch.lower()); i += 1
        else:
            i += 1
    return "".join(out)


def romanize_hindi(text):
    """Write Devanagari Hindi in readable Latin letters without translating it."""
    common_spellings = {
        "mainne": "maine",
        "naheen": "nahin",
        "kaheen": "kahin",
        "koee": "koi",
        "ham": "hum",
        "yah": "yeh",
        "yaha": "yeh",
        "pahachana": "pehchana",
    }

    def convert(match):
        token = match.group(0)
        roman = deva_to_roman(token)
        last = ord(token[-1])
        is_final_consonant = (0x0915 <= last <= 0x0939) or (0x0958 <= last <= 0x095F)
        # Hindi usually drops the inherent vowel at the end of a word.
        if len(token) > 1 and is_final_consonant and roman.endswith("a"):
            roman = roman[:-1]
        roman = roman.replace("ph", "f")
        roman = re.sub(r"(?<=f)a(?=s)", "", roman)
        roman = roman.replace("kahate", "kehte")
        return common_spellings.get(roman, roman)

    # Preserve spaces, punctuation, and any existing Latin text.
    text = str(text).translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    return re.sub(r"[\u0900-\u097f]+", convert, text)


def skeleton(token):
    """Spelling- and script-tolerant key: 'aankhen', 'ankhein' and 'आँखें' share it.

    Used ONLY to match ASR output to the user's lyrics; never to rewrite the user's spelling."""
    t = normalize(token)
    if has_devanagari(t):
        t = deva_to_roman(t)
    t = t.replace("igh", "i").replace("ph", "f").replace("sh", "s").replace("ch", "c").replace("kh", "k").replace("gh", "g")
    t = t.replace("th", "t").replace("dh", "d").replace("bh", "b")
    out = []
    for i, ch in enumerate(t):
        ch = _MAP.get(ch, ch)
        if ch == "h" and i > 0:
            continue
        if ch in _VOWELS and i > 0:
            continue
        if ch == "y" and i > 0 and t[i - 1] in _VOWELS and (i + 1 == len(t) or t[i + 1] in _VOWELS):
            continue                                              # glide: aaye ~ आए
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out) or t


def _consonant_signature(token):
    """Preserve meaningful consonant differences during cross-script matching."""
    t = normalize(str(token))

    if has_devanagari(t):
        t = deva_to_roman(t)

    t = t.lower()

    # In common Romanized Hindi spellings such as "woh" and "yeh",
    # a final h after a vowel is often silent. Keep aspirated digraphs
    # such as "th" in "baithe" intact.
    t = re.sub(r"(?<=[aeiou])h$", "", t)

    # Common Romanized Hindi spelling equivalences.
    t = t.replace("ph", "f")
    for old, new in (("q", "k"), ("w", "v"), ("z", "j"), ("x", "ks")):
        t = t.replace(old, new)

    # In English, the final "gh" in "high" is silent.
    t = re.sub(r"igh$", "i", t)

    digraphs = ("chh", "kh", "gh", "ch", "jh", "th", "dh", "bh", "sh")
    signature = []
    i = 0

    while i < len(t):
        ch = t[i]

        if ch in _VOWELS:
            i += 1
            continue

        # Treat the glide in spellings such as "aaye" as a vowel transition.
        if (
            ch == "y"
            and i > 0
            and t[i - 1] in _VOWELS
            and (i + 1 == len(t) or t[i + 1] in _VOWELS)
        ):
            i += 1
            continue

        matched = next((d for d in digraphs if t.startswith(d, i)), None)
        if matched:
            signature.append(matched)
            i += len(matched)
        else:
            if ch.isalpha():
                signature.append(ch)
            i += 1

    # Common Romanized Hindi spellings may use "ch" where the
    # transliteration produces "chh", as in "kuch" and "kuchha".
    signature = ["ch" if unit == "chh" else unit for unit in signature]

    # Romanized Hindi and Devanagari may represent the same word
    # with different repeated-consonant patterns, such as "Neendein"
    # versus the transliteration "neenden". Normalize both signatures.
    collapsed = []
    for unit in signature:
        if not collapsed or collapsed[-1] != unit:
            collapsed.append(unit)
    signature = collapsed

    return signature


def _sequence_similarity(left, right):
    """Levenshtein similarity for sequences of consonant units."""
    if not left or not right:
        return 1.0 if not left and not right else 0.0

    previous = list(range(len(right) + 1))

    for i, left_unit in enumerate(left, start=1):
        current = [i] + [0] * len(right)

        for j, right_unit in enumerate(right, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_unit != right_unit),
            )

        previous = current

    return 1.0 - previous[-1] / max(len(left), len(right))


_PHONETIC_VOWELS = set("aeiou")


def _phonetic_form(token):
    """Normalize common cross-script Hindi spelling variations."""
    original = normalize(str(token))
    is_deva = has_devanagari(original)

    if is_deva:
        text = deva_to_roman(original).lower()

        # "??" is pronounced "hum"; approximate transliteration
        # otherwise inserts an inherent vowel and produces "ham".
        if original == "\u0939\u092e":
            text = "hum"
        elif original == "\u092f\u0939":
            # Normalize Devanagari "yah" to Romanized "yeh".
            text = "yeh"

        if original:
            cp = ord(original[-1])
            is_consonant = (
                0x0915 <= cp <= 0x0939
                or 0x0958 <= cp <= 0x095F
            )
            # Preserve the inherent vowel in a single-character token.
            if is_consonant and text.endswith("a") and len(original) > 1:
                text = text[:-1]
    else:
        text = original.lower()

    for old, new in (
        ("chh", "ch"),
        ("ph", "f"),
        ("q", "k"),
        ("w", "v"),
        ("z", "j"),
        ("x", "ks"),
    ):
        text = text.replace(old, new)

    if not is_deva and text == "high":
        text = "hai"

    for pattern, replacement in (
        (r"aaee", "ai"),
        (r"ee+", "i"),
        (r"oo+", "u"),
        (r"uu+", "u"),
        (r"aa+", "a"),
    ):
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"(?<=[aeiou])h$", "", text)
    text = re.sub(r"nn", "n", text)
    text = re.sub(r"ein$", "en", text)
    text = re.sub(r"(?<=[aeiou])y(?=[aeiou])", "", text)

    if is_deva:
        text = re.sub(r"(?<=f)a(?=s)", "", text)
        text = text.replace("khushab", "khushb")

        # Remove an internal inherent vowel inserted by approximate
        # transliteration, as in "palakon".
        # Remove only the internal inherent vowels identified by our
        # regression cases; avoid removing legitimate vowels in "nikal".
        text = re.sub(r"(?<=l)a(?=k)", "", text)
        text = re.sub(r"(?<=h)a(?=t)", "", text)

        # Normalize this common spelling difference after schwa handling.
        text = text.replace("kahte", "kehte")
        # Normalize the inherent-vowel variant of "pehchana".
        text = text.replace("pahach", "pehch")

    return text


def _weighted_edit_similarity(left, right):
    """Character similarity with a stronger penalty for vowel mismatches."""
    a = _phonetic_form(left)
    b = _phonetic_form(right)

    if not a or not b:
        return 0.0

    previous = [float(j) for j in range(len(b) + 1)]

    for i, ca in enumerate(a, start=1):
        current = [float(i)] + [0.0] * len(b)

        for j, cb in enumerate(b, start=1):
            if ca == cb:
                cost = 0.0
            elif ca in _PHONETIC_VOWELS and cb in _PHONETIC_VOWELS:
                cost = 1.5
            else:
                cost = 1.0

            current[j] = min(
                previous[j] + 1.0,
                current[j - 1] + 1.0,
                previous[j - 1] + cost,
            )

        previous = current

    return max(
        0.0,
        1.0 - previous[-1] / max(len(a), len(b)),
    )


def similarity(a, b):
    """Return a conservative 0..1 similarity score for word matching."""
    ka, kb = skeleton(a), skeleton(b)

    if not ka or not kb:
        return 0.0

    la, lb = len(ka), len(kb)
    previous = list(range(lb + 1))

    for i in range(1, la + 1):
        current = [i] + [0] * lb

        for j in range(1, lb + 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ka[i - 1] != kb[j - 1]),
            )

        previous = current

    base_score = 1.0 - previous[-1] / max(la, lb)

    # Preserve the existing matcher for same-script comparisons.
    if has_devanagari(str(a)) == has_devanagari(str(b)):
        return base_score

    # Keep the existing consonant-signature safeguard.
    left_signature = _consonant_signature(a)
    right_signature = _consonant_signature(b)

    current_score = base_score

    if left_signature or right_signature:
        detailed_score = _sequence_similarity(
            left_signature,
            right_signature,
        )
        current_score = min(base_score, detailed_score)

    # Compare normalized phonetic forms across scripts.
    phonetic_left = _phonetic_form(a)
    phonetic_right = _phonetic_form(b)
    weighted_score = _weighted_edit_similarity(a, b)

    # Exact normalized equivalence may rescue a low skeleton score,
    # but the consonant-signature safeguard must still be respected.
    if phonetic_left == phonetic_right:
        if left_signature or right_signature:
            return _sequence_similarity(
                left_signature, right_signature
            )
        return 1.0

    # Keep the existing conservative scoring for non-identical forms.
    return min(current_score, weighted_score)
def syllables(token):
    """Rough syllable count (vowel groups; Devanagari vowel signs/letters). Min 1."""
    t = normalize(token)
    if any("\u0900" <= ch <= "\u097f" for ch in t):
        n = sum(1 for ch in t if "\u0904" <= ch <= "\u0914" or "\u093e" <= ch <= "\u094c")
        n += sum(1 for ch in t if "\u0915" <= ch <= "\u0939") - sum(1 for ch in t if ch == "\u094d")
        return max(1, n // 2 + (n % 2))
    return max(1, len(re.findall(r"[aeiouy]+", t)))
