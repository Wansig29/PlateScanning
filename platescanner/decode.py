"""Database-aware plate decoding.

The OCR gives, for every character position, a probability for each possible
character. Taking only the likeliest character throws away exactly what is
needed to tell "ABC1234" from "A8C1234" when the camera is unsure. Here the
whole distribution is scored against the registered plates, so a plate that
is one doubtful character away from a registered one is resolved by the
evidence instead of by a coin flip.

Two competing explanations are weighed (Bayes): the vehicle carries one of
the N registered plates, or it carries some other valid plate (a visitor).
A registered plate is accepted only when it clearly wins, differs from the
plain read in at most a couple of characters, every character it changes was
a plausible alternative to the OCR itself (a visitor must never be turned
into a registered vehicle just because their plates are close), and (softly)
matches the vehicle's colour.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from . import plates

log = logging.getLogger(__name__)

_FLOOR = 1e-4  # no character is ever impossible: the OCR is overconfident


# --- distributions -------------------------------------------------------------

def align_slots(dist: np.ndarray, slots: int, alphabet: str, pad_char: str) -> np.ndarray:
    """Pad a (positions, alphabet) distribution with 'blank' positions up to `slots`."""
    if dist.shape[0] >= slots:
        return dist[:slots]
    blank = np.zeros((slots - dist.shape[0], dist.shape[1]), dist.dtype)
    blank[:, alphabet.index(pad_char)] = 1.0
    return np.vstack([dist, blank])


def temper(dist: np.ndarray, temperature: float) -> np.ndarray:
    """Soften an overconfident distribution (temperature > 1) and renormalise each position."""
    p = np.clip(dist.astype(np.float64), _FLOOR, None) ** (1.0 / max(temperature, 1e-3))
    return p / p.sum(axis=-1, keepdims=True)


def evidence_weight(reads: int) -> float:
    """How many independent reads `reads` frames are worth.

    Frames of one passing vehicle share the same blur, angle and glare, so
    their errors are correlated: the 4th frame adds much less than the 1st.
    """
    return min(3.0, 1.0 + 0.5 * (max(reads, 1) - 1))


def combine(sum_logp: np.ndarray, reads: int, temperature: float) -> np.ndarray:
    """One distribution out of the accumulated log-probabilities of `reads` frames."""
    mean = sum_logp / max(reads, 1)
    logp = evidence_weight(reads) * mean / max(temperature, 1e-3)
    logp -= logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=-1, keepdims=True)


def log_dist(dist: np.ndarray) -> np.ndarray:
    return np.log(np.clip(dist.astype(np.float64), _FLOOR, None))


# --- colour --------------------------------------------------------------------

_COLOUR_WORDS = (
    ("white", ("white", "pearl", "cream", "ivory")),
    ("grey", ("silver", "grey", "gray", "gunmetal", "titanium", "beige", "champagne")),
    ("black", ("black", "charcoal", "dark")),
    ("red", ("red", "maroon", "crimson", "burgundy", "wine")),
    ("orange", ("orange",)),
    ("yellow", ("yellow", "gold")),
    ("green", ("green", "teal", "olive")),
    ("blue", ("blue", "navy", "cyan", "aqua")),
    ("purple", ("purple", "violet", "magenta", "pink")),
    ("brown", ("brown", "bronze", "tan")),
)
# Families the camera can plausibly confuse (shade, glare, white balance).
_NEAR = {frozenset(p) for p in (("white", "grey"), ("grey", "black"), ("red", "orange"), ("red", "brown"),
                                ("orange", "yellow"), ("orange", "brown"), ("yellow", "brown"),
                                ("green", "blue"), ("blue", "purple"), ("red", "purple"))}


def colour_family(text: str | None) -> str | None:
    """"Dark grey", "SILVER", "Maroon" -> a coarse family, or None if unknown."""
    t = (text or "").lower()
    for family, words in _COLOUR_WORDS:
        if any(re.search(rf"\b{w}", t) for w in words):
            return family
    return None


def colours_agree(seen: str | None, registered: str | None) -> bool | None:
    """True/False, or None when either colour is unknown (no evidence either way)."""
    a, b = colour_family(seen), colour_family(registered)
    if a is None or b is None:
        return None
    return a == b or frozenset((a, b)) in _NEAR


# --- lexicon -------------------------------------------------------------------

@dataclass
class Candidate:
    plate: str            # as registered (normalised: letters and digits only)
    posterior: float
    changes: int          # characters that differ from the plain read
    weakest: float = 1.0  # the OCR's probability for the least likely of those changed characters
    colour: str | None = None


@dataclass
class Decoding:
    best: Candidate | None = None       # the winning registered plate, if any is plausible
    runner_up: Candidate | None = None
    visitor: float = 0.0                # posterior that the plate is not a registered one
    text: str = ""                      # the plain (per-position likeliest) read
    accepted: bool = False              # best is trustworthy enough to replace the plain read
    notes: list[str] = field(default_factory=list)


def valid_string_count(layouts: list[str]) -> float:
    """How many plate strings the accepted layouts allow (letters 26, digits 10)."""
    total = sum(26.0 ** lay.count("L") * 10.0 ** lay.count("D") for lay in layouts)
    return total or 36.0 ** 7


class PlateLexicon:
    """The registered plates, scored against an OCR distribution."""

    def __init__(self, alphabet: str, pad_char: str, slots: int, layouts: list[str], *,
                 accept: float = 0.90, max_changes: int = 2, registered_prior: float = 0.7,
                 colour_penalty: float = 0.3, min_char_prob: float = 0.10):
        self.alphabet, self.pad_char, self.slots = alphabet, pad_char, slots
        self.accept, self.max_changes, self.min_char_prob = accept, max_changes, min_char_prob
        self.registered_prior = min(max(registered_prior, 0.01), 0.99)
        self.colour_penalty = colour_penalty
        self.valid_strings = valid_string_count(layouts)
        self._index = {c: i for i, c in enumerate(alphabet)}
        self._pad = self._index[pad_char]
        self._codes = np.zeros((0, slots), np.int16)
        self._plates: list[str] = []
        self._colours: list[str | None] = []
        self._stamp: tuple | None = None

    def __len__(self) -> int:
        return len(self._plates)

    # -- loading
    def refresh(self, conn) -> None:
        """Reload when the vehicles or violations changed since the last load (cheap check)."""
        stamp = (tuple(conn.execute("SELECT count(*), max(updated_at) FROM vehicles").fetchone()),
                 tuple(conn.execute("SELECT count(*), max(updated_at) FROM violations").fetchone()))
        if stamp == self._stamp:
            return
        self._stamp = stamp
        colours: dict[str, str | None] = {}
        for r in conn.execute("SELECT plate, details_json FROM vehicles"):
            try:
                colour = (json.loads(r["details_json"] or "{}") or {}).get("color")
            except (ValueError, TypeError, AttributeError):
                colour = None
            norm = plates.normalize(r["plate"])
            colours[norm] = colours.get(norm) or colour
        # A plate with a violation but no vehicle record must be catchable too.
        for r in conn.execute("SELECT DISTINCT plate FROM violations WHERE plate IS NOT NULL"):
            colours.setdefault(plates.normalize(r["plate"]), None)
        self.load(colours)

    def load(self, registered: dict[str, str | None]) -> None:
        """registered: normalised plate -> registered colour (or None)."""
        rows, names, cols = [], [], []
        for plate, colour in registered.items():
            if not plate or len(plate) > self.slots or any(c not in self._index for c in plate):
                continue
            rows.append([self._index[c] for c in plate] + [self._pad] * (self.slots - len(plate)))
            names.append(plate)
            cols.append(colour)
        self._codes = np.array(rows, np.int16).reshape(-1, self.slots)
        self._plates, self._colours = names, cols

    # -- decoding
    def read_text(self, dist: np.ndarray) -> str:
        idx = dist.argmax(axis=-1)
        return "".join(self.alphabet[i] for i in idx).rstrip(self.pad_char)

    def decode(self, dist: np.ndarray, seen_colour: str | None = None) -> Decoding:
        """dist: (slots, alphabet), each row summing to 1 (see `combine`)."""
        text = self.read_text(dist)
        out = Decoding(text=text)
        n = len(self._plates)
        if n == 0:
            return out
        slot = np.arange(self.slots)
        lik = np.exp(np.log(np.clip(dist, 1e-12, None))[slot, self._codes].sum(axis=1))  # q(plate)

        # P(registered plate i | image) ∝ prior_in / N * q(i)
        # P(some other valid plate | image) ∝ prior_out / valid_strings * (1 - sum of q over registered)
        w = self.registered_prior / n * lik
        for i, colour in enumerate(self._colours):
            if lik[i] > 1e-9 and colours_agree(seen_colour, colour) is False:
                w[i] *= self.colour_penalty
        visitor = (1.0 - self.registered_prior) / self.valid_strings * max(0.0, 1.0 - float(lik.sum()))
        total = float(w.sum()) + visitor
        if total <= 0:
            return out
        post = w / total
        out.visitor = visitor / total

        read_codes = dist.argmax(axis=-1)
        cands = []
        for i in np.argsort(post)[::-1][:2]:
            codes = self._codes[i]
            differ = codes != read_codes
            weakest = float(dist[slot[differ], codes[differ]].min()) if differ.any() else 1.0
            cands.append(Candidate(self._plates[i], float(post[i]), int(differ.sum()), weakest,
                                   self._colours[i]))
        out.best = cands[0]
        out.runner_up = cands[1] if len(cands) > 1 else None
        b = out.best
        out.accepted = (b.posterior >= self.accept and b.changes <= self.max_changes
                        and b.weakest >= self.min_char_prob)
        if b.posterior >= self.accept and not out.accepted:
            out.notes.append(f"{b.plate} fits the evidence but differs from the read {text} in "
                             f"{b.changes} character(s), the least likely {b.weakest:.0%}: not trusted")
        return out
