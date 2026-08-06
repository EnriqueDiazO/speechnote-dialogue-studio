from __future__ import annotations

from collections import Counter
from pathlib import Path

from dialogue_studio.pronunciation import PronunciationEngine
from dialogue_studio.pronunciation.corpus import load_pronunciation_corpus

CORPUS_ROOT = Path("tests/fixtures/pronunciation")


def test_spanish_corpus_meets_curated_discipline_minimums() -> None:
    snapshot = load_pronunciation_corpus(CORPUS_ROOT)
    approved = [case for case in snapshot.approved if case.language == "es"]
    counts = Counter(case.category for case in approved)
    assert len(approved) >= 72
    assert counts >= Counter(
        {
            "basic_arithmetic": 6,
            "algebra": 10,
            "calculus": 14,
            "linear_algebra": 8,
            "probability_statistics": 10,
            "machine_learning": 8,
            "operator_theory": 6,
            "singular_integrals": 5,
            "edge_cases": 5,
        }
    )
    assert {case.profile for case in approved} == {
        "concise",
        "classroom",
        "explicit",
        "symbolic",
    }


def test_required_spanish_expressions_and_unapproved_names_are_present() -> None:
    snapshot = load_pronunciation_corpus(CORPUS_ROOT)
    approved_written = {
        case.written_text for case in snapshot.approved if case.language == "es"
    }
    for expression in (
        "$x^2$",
        "$x_i^2$",
        "$\\frac{dy}{dx}$",
        "$\\iint_D f(x,y)\\,dx\\,dy$",
        "$P(A\\mid B)$",
        "$\\theta_{t+1}=\\theta_t-\\eta\\nabla L(\\theta_t)$",
        "$\\mathcal{B}(H)$",
        "$P_\\Gamma^+$",
        "pipeline",
        "https://example.org/pi",
        "`x_i`",
    ):
        assert expression in approved_written
    candidates = {
        case.written_text for case in snapshot.candidates if case.language == "es"
    }
    assert candidates == {"Haseman", "Fredholm", "Wiener–Hopf", "Mellin", "Calkin"}


def test_curated_spanish_outputs_match_reviewed_expectations() -> None:
    snapshot = load_pronunciation_corpus(CORPUS_ROOT)
    engine = PronunciationEngine()
    for case in snapshot.approved:
        if case.language != "es":
            continue
        result = engine.transform(
            case.written_text,
            profile=case.pronunciation_profile(),
        )
        message = (
            f"CASE: {case.case_id}\nEXPECTED: {case.expected_spoken_text}"
            f"\nACTUAL: {result.spoken_text}"
        )
        if case.assertion_mode == "exact":
            assert result.spoken_text == case.expected_spoken_text, message
        elif case.assertion_mode == "semantic":
            cursor = 0
            for anchor in case.semantic_anchors:
                position = result.spoken_text.find(anchor, cursor)
                assert position >= 0, f"{message}\nMISSING ANCHOR: {anchor}"
                cursor = position + len(anchor)
        assert tuple(warning.code for warning in result.warnings) == (
            case.expected_warning_codes
        ), message
        assert result.unsupported_fragments == case.expected_unsupported_fragments, message
        assert not any(
            fragment in result.spoken_text for fragment in case.forbidden_fragments
        ), message
