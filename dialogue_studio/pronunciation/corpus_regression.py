"""Deterministic regression assertions for approved pronunciation cases."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .corpus import PronunciationCorpusCase, PronunciationCorpusSnapshot
from .engine import PronunciationEngine
from .models import PronunciationResult


class CorpusRegressionError(AssertionError):
    """An approved corpus case no longer matches its reviewed contract."""


@dataclass(frozen=True)
class CorpusRegressionSummary:
    executed_case_ids: tuple[str, ...]

    @property
    def executed(self) -> int:
        return len(self.executed_case_ids)


def regression_cases(
    snapshot: PronunciationCorpusSnapshot,
) -> tuple[PronunciationCorpusCase, ...]:
    """Return only canonical cases, in stable case-id order."""
    return tuple(sorted(snapshot.approved, key=lambda case: case.case_id))


def _first_difference(expected: str, actual: str) -> str:
    limit = min(len(expected), len(actual))
    position = next(
        (index for index in range(limit) if expected[index] != actual[index]),
        limit,
    )
    if expected == actual:
        return "none"
    start = max(0, position - 24)
    end = position + 24
    expected_char = repr(expected[position]) if position < len(expected) else "<end>"
    actual_char = repr(actual[position]) if position < len(actual) else "<end>"
    return (
        f"index {position}: expected {expected_char}, actual {actual_char}\n"
        f"expected context: {expected[start:end]!r}\n"
        f"actual context:   {actual[start:end]!r}"
    )


def _failure_message(
    case: PronunciationCorpusCase,
    result: PronunciationResult,
    detail: str,
) -> str:
    return (
        f"CASE: {case.case_id}\n"
        f"PROFILE: {case.profile}\n"
        f"WRITTEN:\n{case.written_text}\n\n"
        f"EXPECTED:\n{case.expected_spoken_text}\n\n"
        f"ACTUAL:\n{result.spoken_text}\n\n"
        f"FIRST DIFFERENCE:\n"
        f"{_first_difference(case.expected_spoken_text, result.spoken_text)}\n\n"
        f"CONTRACT FAILURE:\n{detail}"
    )


def _ordered_missing(text: str, anchors: Iterable[str]) -> str | None:
    cursor = 0
    for anchor in anchors:
        position = text.find(anchor, cursor)
        if position < 0:
            return anchor
        cursor = position + len(anchor)
    return None


def _sequence_missing(actual: tuple[str, ...], expected: Iterable[str]) -> str | None:
    cursor = 0
    for item in expected:
        try:
            position = actual.index(item, cursor)
        except ValueError:
            return item
        cursor = position + 1
    return None


def assert_corpus_result(
    case: PronunciationCorpusCase,
    result: PronunciationResult,
) -> None:
    """Apply every reviewed assertion attached to one approved case."""
    if case.status != "approved":
        raise ValueError(f"{case.case_id}: sólo los casos approved fijan regresiones")

    if case.assertion_mode == "exact" and result.spoken_text != case.expected_spoken_text:
        raise CorpusRegressionError(_failure_message(case, result, "exact output mismatch"))

    if case.assertion_mode == "semantic":
        missing = _ordered_missing(result.spoken_text, case.semantic_anchors)
        if missing is not None:
            raise CorpusRegressionError(
                _failure_message(
                    case,
                    result,
                    f"semantic anchor missing or out of order: {missing!r}",
                )
            )

    actual_warning_codes = tuple(warning.code for warning in result.warnings)
    if actual_warning_codes != case.expected_warning_codes:
        raise CorpusRegressionError(
            _failure_message(
                case,
                result,
                "warning codes mismatch: "
                f"expected {case.expected_warning_codes!r}, actual {actual_warning_codes!r}",
            )
        )

    if result.unsupported_fragments != case.expected_unsupported_fragments:
        raise CorpusRegressionError(
            _failure_message(
                case,
                result,
                "unsupported fragments mismatch: "
                f"expected {case.expected_unsupported_fragments!r}, "
                f"actual {result.unsupported_fragments!r}",
            )
        )

    forbidden = next(
        (
            fragment
            for fragment in case.forbidden_fragments
            if fragment in result.spoken_text
        ),
        None,
    )
    if forbidden is not None:
        raise CorpusRegressionError(
            _failure_message(case, result, f"forbidden fragment present: {forbidden!r}")
        )

    if case.applied_rule_ids:
        actual_rule_ids = tuple(rule.rule_id for rule in result.applied_rules)
        missing = _sequence_missing(actual_rule_ids, case.applied_rule_ids)
        if missing is not None:
            raise CorpusRegressionError(
                _failure_message(
                    case,
                    result,
                    f"applied rule ID missing or out of order: {missing!r}",
                )
            )


def execute_corpus_case(
    case: PronunciationCorpusCase,
    *,
    engine: PronunciationEngine | None = None,
) -> PronunciationResult:
    selected_engine = engine or PronunciationEngine()
    result = selected_engine.transform(
        case.written_text,
        profile=case.pronunciation_profile(),
    )
    assert_corpus_result(case, result)
    return result


def run_pronunciation_regression(
    snapshot: PronunciationCorpusSnapshot,
    *,
    engine: PronunciationEngine | None = None,
) -> CorpusRegressionSummary:
    """Execute all and only approved cases."""
    selected_engine = engine or PronunciationEngine()
    executed: list[str] = []
    for case in regression_cases(snapshot):
        execute_corpus_case(case, engine=selected_engine)
        executed.append(case.case_id)
    return CorpusRegressionSummary(executed_case_ids=tuple(executed))
