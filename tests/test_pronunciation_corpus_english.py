from __future__ import annotations

from pathlib import Path

from dialogue_studio.pronunciation.corpus import load_pronunciation_corpus

CORPUS_ROOT = Path("tests/fixtures/pronunciation")


def test_english_corpus_is_representative_and_uses_all_profiles() -> None:
    snapshot = load_pronunciation_corpus(CORPUS_ROOT)
    approved = [case for case in snapshot.approved if case.language == "en"]
    assert len(approved) >= 24
    assert {case.profile for case in approved} == {
        "concise",
        "classroom",
        "explicit",
        "symbolic",
    }
    written = {case.written_text for case in approved}
    for required in (
        "$x^2$",
        "$x_i$",
        "$\\frac{dy}{dx}$",
        "$\\int_a^b f(x)\\,dx$",
        "$\\sum_{i=1}^{n}a_i$",
        "$\\lim_{x\\to a}f(x)$",
        "$P(A\\mid B)$",
        "$\\vec{x}$",
        "MSE",
        "ReLU",
        "$\\unknowncommand{x}$",
        "https://example.org/pi",
        "`x_i`",
    ):
        assert required in written
