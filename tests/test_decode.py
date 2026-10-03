"""Database-aware decoding: the OCR's full character probabilities scored against registered plates."""
import numpy as np

from platescanner import decode
from platescanner.vision.tracker import Track

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
LAYOUTS = ["LLLDDDD", "LLLDDD"]
SLOTS = 10


def dist_for(text: str, sure: float = 0.97, unsure: dict[int, dict[str, float]] | None = None) -> np.ndarray:
    """What the OCR would output for `text`: `sure` on each character, the rest spread evenly.
    unsure: position -> {char: probability} to override a position's distribution."""
    rows = np.zeros((SLOTS, len(ALPHABET)))
    for i in range(SLOTS):
        ch = text[i] if i < len(text) else "_"
        rows[i, :] = (1 - sure) / (len(ALPHABET) - 1)
        rows[i, ALPHABET.index(ch)] = sure
    for i, probs in (unsure or {}).items():
        rows[i, :] = 0.001 / (len(ALPHABET) - len(probs))
        for ch, p in probs.items():
            rows[i, ALPHABET.index(ch)] = p
        rows[i] /= rows[i].sum()
    return rows


def lexicon(registered: dict[str, str | None], **kw) -> decode.PlateLexicon:
    lex = decode.PlateLexicon(ALPHABET, "_", SLOTS, LAYOUTS, **kw)
    lex.load(registered)
    return lex


REGISTERED = {"NBC1234": "White", "ABC1234": "Red", "XYZ789": None, "DEF5678": "Black"}


def test_clean_read_of_registered_plate_is_accepted():
    dec = lexicon(REGISTERED).decode(dist_for("NBC1234"))
    assert dec.accepted and dec.best.plate == "NBC1234" and dec.best.changes == 0


def test_doubtful_character_is_resolved_by_the_database():
    # Position 0 is N or W, nearly a coin flip; only NBC1234 is registered.
    d = dist_for("NBC1234", unsure={0: {"W": 0.5, "N": 0.45}})
    assert lexicon(REGISTERED).read_text(d)[0] == "W"  # the plain read is wrong
    dec = lexicon(REGISTERED).decode(d)
    assert dec.accepted and dec.best.plate == "NBC1234" and dec.best.changes == 1


def test_two_registered_plates_that_both_fit_are_not_guessed():
    lex = lexicon({"NBC1234": None, "WBC1234": None})
    dec = lex.decode(dist_for("NBC1234", unsure={0: {"W": 0.5, "N": 0.5}}))
    assert not dec.accepted
    assert {dec.best.plate, dec.runner_up.plate} == {"NBC1234", "WBC1234"}


def test_confident_unregistered_plate_stays_a_visitor():
    dec = lexicon(REGISTERED).decode(dist_for("QWE4567"))
    assert not dec.accepted and dec.visitor > 0.9


def test_confident_read_is_not_overridden_by_a_nearby_registered_plate():
    # The OCR is sure of "NBC1239"; "NBC1234" is registered, but it was never a plausible reading.
    dec = lexicon(REGISTERED).decode(decode.temper(dist_for("NBC1239", sure=0.99), 2.0))
    assert dec.best.plate == "NBC1234" and dec.best.weakest < 0.10
    assert not dec.accepted and dec.notes


def test_a_registered_plate_many_characters_off_is_never_accepted():
    # Everything doubtful: even if it scores well, more than max_changes differences are refused.
    unsure = {i: {"X": 0.45, "ABC1234"[i]: 0.35} for i in range(7)}
    dec = lexicon({"ABC1234": None}, max_changes=2).decode(dist_for("XXXXXXX", unsure=unsure))
    assert dec.best.plate == "ABC1234" and dec.best.changes > 2 and not dec.accepted


def test_clearly_different_colour_counts_against_a_candidate():
    lex = lexicon({"NBC1234": "White", "WBC1234": "Black"})
    d = dist_for("NBC1234", unsure={0: {"W": 0.5, "N": 0.5}})
    plain = lex.decode(d)
    seen_white = lex.decode(d, "White")
    assert seen_white.best.plate == "NBC1234" and seen_white.best.posterior > plain.best.posterior
    assert lex.decode(d, "Black").best.plate == "WBC1234"


def test_colour_families():
    assert decode.colours_agree("Silver / grey", "Gray") is True
    assert decode.colours_agree("White", "Silver") is True      # shade and glare
    assert decode.colours_agree("Red", "Maroon") is True
    assert decode.colours_agree("Red", "Blue") is False
    assert decode.colours_agree("Red", None) is None
    assert decode.colours_agree(None, "Red") is None
    assert decode.colours_agree("Teal", "Pink") is False


def test_unknown_or_unusable_registered_plates_are_skipped():
    lex = lexicon({"": None, "TOOLONGPLATE123": None, "AB-1": None, "ABC1234": None})
    assert len(lex) == 1


def test_empty_database_decodes_nothing():
    assert lexicon({}).decode(dist_for("ABC1234")).best is None


def test_more_slots_in_a_shorter_model_are_padded():
    short = np.random.default_rng(0).dirichlet(np.ones(len(ALPHABET)), size=9)
    out = decode.align_slots(short, 10, ALPHABET, "_")
    assert out.shape == (10, len(ALPHABET)) and out[9, ALPHABET.index("_")] == 1.0


def test_track_combines_reads_and_temperature_softens():
    t = Track(1, (0, 0, 10, 5), 0.0, 0.0)
    for _ in range(3):
        t.add_vote("NBC1234", "NBC1234", 0.9, dist=dist_for("NBC1234", unsure={0: {"N": 0.6, "W": 0.4}}))
    d = t.distribution(1.0)
    assert d.shape == (SLOTS, len(ALPHABET)) and np.allclose(d.sum(axis=-1), 1)
    assert d[0, ALPHABET.index("N")] > d[0, ALPHABET.index("W")]
    # Repeated frames share their errors, so they count for less than independent reads.
    assert decode.evidence_weight(8) <= 3.0 and decode.evidence_weight(1) == 1.0
    soft = t.distribution(3.0)
    assert soft[0, ALPHABET.index("N")] < d[0, ALPHABET.index("N")]
