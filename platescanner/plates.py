"""Plate text normalization, OCR-confusion handling and layout checks."""
from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^A-Z0-9]")

# Characters OCR commonly confuses, folded to one representative so that
# "ABC 1O34" and "ABC 1034" produce the same lookup key.
_FOLD = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
})

# Position-aware coercion when a layout expects a letter / digit.
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
_LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
                    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}


def normalize(text: str | None) -> str:
    """Uppercase and strip everything but A-Z / 0-9."""
    return _NON_ALNUM.sub("", (text or "").upper())


def plate_key(text: str | None) -> str:
    """Confusion-folded key used for database matching."""
    return normalize(text).translate(_FOLD)


def is_repeated_pattern(text: str | None) -> bool:
    """True for a read that is one character over and over once look-alikes are folded,
    e.g. "II 1111" or "OOO 000". Gate grilles, fences and tiles read like this; real plates don't."""
    return len(set(plate_key(text))) <= 1


def coerce(text: str, layout: str) -> str | None:
    """Force `text` into `layout` (e.g. "LLLDDDD"), fixing confusable characters.

    Returns None if the length differs or a character can't be coerced.
    """
    text = normalize(text)
    if len(text) != len(layout):
        return None
    out = []
    for ch, kind in zip(text, layout):
        if kind == "L":
            if ch.isalpha():
                out.append(ch)
            elif ch in _DIGIT_TO_LETTER:
                out.append(_DIGIT_TO_LETTER[ch])
            else:
                return None
        else:
            if ch.isdigit():
                out.append(ch)
            elif ch in _LETTER_TO_DIGIT:
                out.append(_LETTER_TO_DIGIT[ch])
            else:
                return None
    return "".join(out)


def best_layout_match(text: str, layouts: list[str]) -> str | None:
    """Coerce into the first layout needing the fewest character changes."""
    raw = normalize(text)
    best: tuple[int, str] | None = None
    for layout in layouts:
        fixed = coerce(raw, layout)
        if fixed is None:
            continue
        changes = sum(a != b for a, b in zip(raw, fixed))
        if best is None or changes < best[0]:
            best = (changes, fixed)
    return best[1] if best else None


def display(text: str | None) -> str:
    """"ABC1234" -> "ABC 1234": a space between the letter and digit groups."""
    t = normalize(text)
    m = re.fullmatch(r"([A-Z]+)(\d+)", t) or re.fullmatch(r"(\d+)([A-Z]+)", t)
    return f"{m.group(1)} {m.group(2)}" if m else t


def within_one_edit(a: str, b: str) -> bool:
    """True if a and b differ by at most one insert, delete or substitution."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b = b, a
    # b is one longer: skip exactly one char of b.
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True
