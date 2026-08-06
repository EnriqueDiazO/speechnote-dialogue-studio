from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dialogue_studio.pronunciation.corpus import (
    PronunciationCorpusCase,
    load_pronunciation_corpus,
)
from dialogue_studio.pronunciation.corpus_regression import (
    CorpusRegressionError,
    assert_corpus_result,
    execute_corpus_case,
    regression_cases,
    run_pronunciation_regression,
)

CORPUS_ROOT = Path("tests/fixtures/pronunciation")
SNAPSHOT = load_pronunciation_corpus(CORPUS_ROOT)


@pytest.mark.parametrize("case", regression_cases(SNAPSHOT), ids=lambda case: case.case_id)
def test_every_approved_case_matches_its_reviewed_contract(
    case: PronunciationCorpusCase,
) -> None:
    execute_corpus_case(case)


def test_candidates_and_deprecated_cases_do_not_enter_regression() -> None:
    selected = {case.case_id for case in regression_cases(SNAPSHOT)}
    excluded = {
        case.case_id for case in (*SNAPSHOT.candidates, *SNAPSHOT.deprecated)
    }
    assert selected.isdisjoint(excluded)
    summary = run_pronunciation_regression(SNAPSHOT)
    assert summary.executed == len(SNAPSHOT.approved)
    assert set(summary.executed_case_ids) == selected


def test_intentional_expected_output_failure_has_a_readable_diff() -> None:
    original = SNAPSHOT.approved[0]
    changed_in_memory = replace(original, expected_spoken_text="lectura incorrecta")
    with pytest.raises(CorpusRegressionError) as captured:
        execute_corpus_case(changed_in_memory)
    message = str(captured.value)
    assert f"CASE: {original.case_id}" in message
    assert f"PROFILE: {original.profile}" in message
    assert "WRITTEN:" in message
    assert "EXPECTED:\nlectura incorrecta" in message
    assert "ACTUAL:" in message
    assert "FIRST DIFFERENCE:\nindex " in message


def test_semantic_anchors_must_appear_in_declared_order() -> None:
    original = next(case for case in SNAPSHOT.approved if case.assertion_mode == "semantic")
    result = execute_corpus_case(original)
    reversed_anchors = replace(original, semantic_anchors=original.semantic_anchors[::-1])
    with pytest.raises(CorpusRegressionError, match="out of order"):
        assert_corpus_result(reversed_anchors, result)


def test_forbidden_fragments_are_enforced() -> None:
    original = next(case for case in SNAPSHOT.approved if case.assertion_mode == "exact")
    result = execute_corpus_case(original)
    forbidden = replace(original, forbidden_fragments=(result.spoken_text,))
    with pytest.raises(CorpusRegressionError, match="forbidden fragment"):
        assert_corpus_result(forbidden, result)
