from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dialogue_studio.pronunciation.corpus import load_pronunciation_corpus
from dialogue_studio.pronunciation.corpus_maintenance import (
    ConfirmationRequired,
    add_candidate,
    corpus_statistics,
    deprecate_case,
    preview_case,
    promote_case,
    validate_corpus,
)
from scripts.pronunciation_corpus import main

CORPUS_ROOT = Path("tests/fixtures/pronunciation")


def _copied_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "pronunciation"
    shutil.copytree(CORPUS_ROOT, root)
    return root


def _candidate_file(tmp_path: Path, *, category: str = "calculus") -> Path:
    source = tmp_path / "candidate.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "case_id": "en-calculus-ui-candidate-999",
                "status": "candidate",
                "language": "en",
                "profile": "classroom",
                "category": category,
                "written_text": "$x^2$",
                "expected_spoken_text": "x squared",
                "assertion_mode": "exact",
                "expected_warning_codes": [],
                "expected_unsupported_fragments": [],
                "semantic_anchors": [],
                "forbidden_fragments": [],
                "applied_rule_ids": ["builtin-math-classroom"],
                "tags": ["ui_export"],
                "notes": "Pendiente de revisión humana.",
                "source_kind": "ui_export",
                "source_reference": "",
                "created_at": "2026-08-05T00:00:00+00:00",
                "updated_at": "2026-08-05T00:00:00+00:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return source


def test_validate_and_statistics_cover_the_checked_in_corpus() -> None:
    snapshot = validate_corpus(CORPUS_ROOT)
    stats = corpus_statistics(snapshot)
    assert stats["total"] == 113
    assert stats["approved"] == 108
    assert stats["by_language"] == {"en": 25, "es": 88}
    assert stats["with_warnings"] == 3


def test_add_candidate_is_atomic_preserves_source_and_rejects_duplicate(
    tmp_path: Path,
) -> None:
    root = _copied_corpus(tmp_path)
    source = _candidate_file(tmp_path)
    original_source = source.read_bytes()
    added = add_candidate(root, source)
    assert source.read_bytes() == original_source
    assert added.status == "candidate"
    snapshot = load_pronunciation_corpus(root)
    assert len(snapshot.candidates) == 6
    assert len(snapshot.approved) == 108
    assert snapshot.manifest.case_counts["total"] == 114
    with pytest.raises(ValueError, match="duplicado"):
        add_candidate(root, source)


def test_candidate_promotion_requires_confirmation_and_keeps_expected_explicit(
    tmp_path: Path,
) -> None:
    root = _copied_corpus(tmp_path)
    source = _candidate_file(tmp_path)
    candidate = add_candidate(root, source)
    preview = preview_case(load_pronunciation_corpus(root), candidate.case_id)
    assert preview.current_spoken_text == "x squared"
    assert preview.case.expected_spoken_text == "x squared"
    with pytest.raises(ConfirmationRequired, match="--confirm"):
        promote_case(root, candidate.case_id, confirm=False)
    assert preview_case(load_pronunciation_corpus(root), candidate.case_id).case.status == (
        "candidate"
    )

    promoted = promote_case(root, candidate.case_id, confirm=True)
    assert promoted.status == "approved"
    snapshot = validate_corpus(root)
    assert promoted.case_id in {case.case_id for case in snapshot.approved}
    assert promoted.expected_spoken_text == "x squared"


def test_deprecation_requires_confirmation_and_retains_traceability(
    tmp_path: Path,
) -> None:
    root = _copied_corpus(tmp_path)
    case_id = "en-algebra-square-001"
    with pytest.raises(ConfirmationRequired, match="--confirm"):
        deprecate_case(root, case_id, confirm=False)
    deprecated = deprecate_case(root, case_id, confirm=True)
    snapshot = load_pronunciation_corpus(root)
    assert deprecated.status == "deprecated"
    assert case_id in {case.case_id for case in snapshot.deprecated}
    assert case_id not in {case.case_id for case in snapshot.approved}


def test_new_category_can_be_added_without_changing_loader_code(tmp_path: Path) -> None:
    root = _copied_corpus(tmp_path)
    source = _candidate_file(tmp_path, category="future_field")
    add_candidate(root, source)
    snapshot = load_pronunciation_corpus(root)
    assert "future_field" in snapshot.manifest.categories
    assert any(case.category == "future_field" for case in snapshot.candidates)


def test_cli_lists_shows_stats_and_prints_preview_before_confirmation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _copied_corpus(tmp_path)
    source = _candidate_file(tmp_path)
    assert main(["--root", str(root), "add-candidate", "--file", str(source)]) == 0
    case_id = "en-calculus-ui-candidate-999"

    assert main(["--root", str(root), "list", "--status", "candidate"]) == 0
    assert case_id in capsys.readouterr().out
    assert main(["--root", str(root), "show", case_id]) == 0
    assert '"source_kind": "ui_export"' in capsys.readouterr().out
    assert main(["--root", str(root), "stats"]) == 0
    assert '"candidate": 6' in capsys.readouterr().out

    assert main(["--root", str(root), "promote", case_id]) == 2
    output = capsys.readouterr()
    assert "CURRENT SPOKEN:\nx squared" in output.out
    assert "requiere --confirm" in output.err
