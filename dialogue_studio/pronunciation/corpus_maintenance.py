"""Safe, deterministic maintenance operations for pronunciation corpora."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..storage import atomic_write_text, deterministic_json
from .corpus import (
    CORPUS_SCHEMA_VERSION,
    PronunciationCorpusCase,
    PronunciationCorpusSnapshot,
    load_pronunciation_corpus,
)
from .corpus_regression import run_pronunciation_regression
from .engine import PronunciationEngine


class ConfirmationRequired(ValueError):
    """A corpus state transition was attempted without explicit confirmation."""


@dataclass(frozen=True)
class CorpusCasePreview:
    case: PronunciationCorpusCase
    current_spoken_text: str
    warning_codes: tuple[str, ...]
    unsupported_fragments: tuple[str, ...]


def validate_corpus(root: Path) -> PronunciationCorpusSnapshot:
    snapshot = load_pronunciation_corpus(root)
    run_pronunciation_regression(snapshot)
    return snapshot


def corpus_statistics(snapshot: PronunciationCorpusSnapshot) -> dict[str, object]:
    return {
        "total": len(snapshot.cases),
        "approved": len(snapshot.approved),
        "candidate": len(snapshot.candidates),
        "deprecated": len(snapshot.deprecated),
        "by_language": dict(sorted(Counter(case.language for case in snapshot.cases).items())),
        "by_profile": dict(sorted(Counter(case.profile for case in snapshot.cases).items())),
        "by_category": dict(sorted(Counter(case.category for case in snapshot.cases).items())),
        "with_warnings": sum(bool(case.expected_warning_codes) for case in snapshot.cases),
        "with_unsupported_fragments": sum(
            bool(case.expected_unsupported_fragments) for case in snapshot.cases
        ),
    }


def write_corpus_report(
    snapshot: PronunciationCorpusSnapshot,
    destination: Path,
) -> Path:
    """Write a reproducible report without replacing an existing review artifact."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"El reporte ya existe: {destination}")
    report = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_version": snapshot.manifest.corpus_version,
        "last_validated_at": snapshot.manifest.last_validated_at,
        "statistics": corpus_statistics(snapshot),
    }
    atomic_write_text(destination, deterministic_json(report))
    return destination


def find_case(
    snapshot: PronunciationCorpusSnapshot, case_id: str
) -> PronunciationCorpusCase:
    try:
        return next(case for case in snapshot.cases if case.case_id == case_id)
    except StopIteration as exc:
        raise ValueError(f"No existe el caso: {case_id}") from exc


def preview_case(
    snapshot: PronunciationCorpusSnapshot, case_id: str
) -> CorpusCasePreview:
    case = find_case(snapshot, case_id)
    result = PronunciationEngine().transform(
        case.written_text,
        profile=case.pronunciation_profile(),
    )
    return CorpusCasePreview(
        case=case,
        current_spoken_text=result.spoken_text,
        warning_codes=tuple(warning.code for warning in result.warnings),
        unsupported_fragments=result.unsupported_fragments,
    )


def _load_candidate_source(path: Path) -> PronunciationCorpusCase:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"El candidato no es un archivo regular: {path}")
    if path.stat().st_size > 1_000_000:
        raise ValueError("El archivo candidato supera 1 MB")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON candidato inválido: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("El candidato debe ser un objeto JSON")
    if "cases" in raw:
        cases = raw["cases"]
        if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
            raise ValueError("El archivo candidato debe contener exactamente un caso")
        raw = cases[0]
    case = PronunciationCorpusCase.from_dict(raw)
    if case.status != "candidate":
        raise ValueError("add-candidate sólo acepta status=candidate")
    return case


def _file_payload(cases: list[PronunciationCorpusCase]) -> dict[str, object]:
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "cases": [case.to_dict() for case in sorted(cases, key=lambda item: item.case_id)],
    }


def _read_case_file(path: Path) -> list[PronunciationCorpusCase]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"El destino no es un archivo regular: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON de destino inválido: {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise ValueError(f"Archivo de corpus inválido: {path}")
    return [
        PronunciationCorpusCase.from_dict(case)
        for case in raw["cases"]
        if isinstance(case, dict)
    ]


def _updated_manifest(
    snapshot: PronunciationCorpusSnapshot,
    cases: list[PronunciationCorpusCase],
) -> dict[str, Any]:
    counts = Counter(case.status for case in cases)
    data = snapshot.manifest.to_dict()
    data["categories"] = sorted(
        {*snapshot.manifest.categories, *(case.category for case in cases)}
    )
    data["case_counts"] = {
        "total": len(cases),
        "approved": counts["approved"],
        "candidate": counts["candidate"],
        "deprecated": counts["deprecated"],
    }
    return data


def _commit_files(
    snapshot: PronunciationCorpusSnapshot,
    *,
    replacements: dict[Path, list[PronunciationCorpusCase]],
    resulting_cases: list[PronunciationCorpusCase],
) -> None:
    for path in sorted(replacements, key=lambda item: str(item)):
        atomic_write_text(path, deterministic_json(_file_payload(replacements[path])))
    atomic_write_text(
        snapshot.root / "manifest.json",
        deterministic_json(_updated_manifest(snapshot, resulting_cases)),
    )
    load_pronunciation_corpus(snapshot.root)


def add_candidate(root: Path, source: Path) -> PronunciationCorpusCase:
    snapshot = load_pronunciation_corpus(root)
    candidate = _load_candidate_source(source)
    if candidate.case_id in snapshot.case_paths:
        raise ValueError(f"case_id duplicado: {candidate.case_id}")
    target = snapshot.root / "candidates" / candidate.language / "ui_candidates.json"
    target_cases = _read_case_file(target)
    resulting_cases = [*snapshot.cases, candidate]
    _commit_files(
        snapshot,
        replacements={target: [*target_cases, candidate]},
        resulting_cases=resulting_cases,
    )
    return candidate


def _transition_case(
    root: Path,
    case_id: str,
    *,
    destination_status: str,
    destination_directory: str,
    confirm: bool,
) -> PronunciationCorpusCase:
    snapshot = load_pronunciation_corpus(root)
    case = find_case(snapshot, case_id)
    if not confirm:
        raise ConfirmationRequired("La operación requiere --confirm")
    if case.status == destination_status:
        raise ValueError(f"{case_id} ya tiene status={destination_status}")
    transitioned = replace(case, status=destination_status)
    transitioned.validate()
    source = snapshot.case_paths[case_id]
    source_cases = [item for item in _read_case_file(source) if item.case_id != case_id]
    target = (
        snapshot.root
        / destination_directory
        / transitioned.language
        / f"{transitioned.category}.json"
    )
    target_cases = [
        item for item in _read_case_file(target) if item.case_id != transitioned.case_id
    ]
    replacements = {source: source_cases, target: [*target_cases, transitioned]}
    if source == target:
        replacements = {source: [*source_cases, transitioned]}
    resulting_cases = [
        transitioned if item.case_id == case_id else item for item in snapshot.cases
    ]
    _commit_files(
        snapshot,
        replacements=replacements,
        resulting_cases=resulting_cases,
    )
    return transitioned


def promote_case(root: Path, case_id: str, *, confirm: bool) -> PronunciationCorpusCase:
    snapshot = load_pronunciation_corpus(root)
    case = find_case(snapshot, case_id)
    if case.status != "candidate":
        raise ValueError("Sólo se puede promover un caso candidate")
    return _transition_case(
        root,
        case_id,
        destination_status="approved",
        destination_directory="approved",
        confirm=confirm,
    )


def deprecate_case(root: Path, case_id: str, *, confirm: bool) -> PronunciationCorpusCase:
    return _transition_case(
        root,
        case_id,
        destination_status="deprecated",
        destination_directory="deprecated",
        confirm=confirm,
    )
