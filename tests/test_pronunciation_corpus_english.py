from __future__ import annotations

from pathlib import Path

from dialogue_studio.pronunciation import PronunciationEngine
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


def test_curated_english_outputs_are_not_spanish_translations() -> None:
    snapshot = load_pronunciation_corpus(CORPUS_ROOT)
    engine = PronunciationEngine()
    for case in snapshot.approved:
        if case.language != "en":
            continue
        result = engine.transform(case.written_text, profile=case.pronunciation_profile())
        context = f"CASE: {case.case_id}\nACTUAL: {result.spoken_text}"
        if case.assertion_mode == "exact":
            assert result.spoken_text == case.expected_spoken_text, context
        assert tuple(warning.code for warning in result.warnings) == (
            case.expected_warning_codes
        ), context
        assert result.unsupported_fragments == case.expected_unsupported_fragments, context
        assert not any(
            fragment in result.spoken_text for fragment in case.forbidden_fragments
        ), context
