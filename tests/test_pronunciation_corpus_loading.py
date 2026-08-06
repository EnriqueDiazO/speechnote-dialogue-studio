from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dialogue_studio.pronunciation.corpus import load_pronunciation_corpus

CORPUS_ROOT = Path("tests/fixtures/pronunciation")


def test_corpus_discovers_versioned_files_without_fixed_file_list() -> None:
    snapshot = load_pronunciation_corpus(CORPUS_ROOT)
    assert snapshot.manifest.corpus_version == "1.0.0"
    assert len(snapshot.candidates) == 5
    assert len(snapshot.approved) == 108
    assert not snapshot.deprecated
    assert {case.written_text for case in snapshot.candidates} == {
        "Haseman",
        "Fredholm",
        "Wiener–Hopf",
        "Mellin",
        "Calkin",
    }
    assert all(not case.expected_spoken_text for case in snapshot.candidates)
    assert all(snapshot.case_paths[case.case_id].is_file() for case in snapshot.cases)


def _copied_corpus(tmp_path: Path) -> Path:
    destination = tmp_path / "pronunciation"
    shutil.copytree(CORPUS_ROOT, destination)
    return destination


def test_empty_corpus_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    for name in ("approved", "candidates", "deprecated"):
        (root / name).mkdir(parents=True)
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    manifest["case_counts"] = {
        "total": 0,
        "approved": 0,
        "candidate": 0,
        "deprecated": 0,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="vacío"):
        load_pronunciation_corpus(root)


def test_corrupt_json_is_rejected_with_its_path(tmp_path: Path) -> None:
    root = _copied_corpus(tmp_path)
    broken = root / "approved" / "en" / "cases.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match=r"JSON de corpus inválido.*cases\.json"):
        load_pronunciation_corpus(root)


def test_unknown_file_schema_version_is_rejected(tmp_path: Path) -> None:
    root = _copied_corpus(tmp_path)
    path = root / "approved" / "en" / "cases.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version de archivo"):
        load_pronunciation_corpus(root)


def test_duplicate_case_id_between_files_is_rejected(tmp_path: Path) -> None:
    root = _copied_corpus(tmp_path)
    source = root / "approved" / "es" / "basic_arithmetic.json"
    duplicate = root / "approved" / "es" / "duplicate.json"
    duplicate.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="case_id duplicado"):
        load_pronunciation_corpus(root)


def test_manifest_counts_must_match_discovered_cases(tmp_path: Path) -> None:
    root = _copied_corpus(tmp_path)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["case_counts"]["total"] += 1
    manifest["case_counts"]["approved"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="case_counts inconsistente"):
        load_pronunciation_corpus(root)
