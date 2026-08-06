"""Portable, download-only corpus candidates built from pronunciation results."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from ..storage import deterministic_json
from .corpus import CORPUS_SCHEMA_VERSION, PronunciationCorpusCase
from .models import PronunciationResult

_PERSONAL_PATH = re.compile(r"(?:/home/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)", re.IGNORECASE)


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def _portable_tags(tags: Iterable[str]) -> tuple[str, ...]:
    normalized = [_normalized(tag) for tag in tags]
    return tuple(dict.fromkeys(tag for tag in normalized if tag))


def build_corpus_candidate(
    result: PronunciationResult,
    *,
    case_id: str,
    category: str,
    tags: Iterable[str] = (),
    notes: str = "",
) -> PronunciationCorpusCase:
    """Create a candidate without project identity, paths, audio, logs or secrets."""
    normalized_notes = _normalized(notes)
    normalized_tags = _portable_tags(tags)
    portability_fields = (
        result.written_text,
        result.spoken_text,
        normalized_notes,
        *normalized_tags,
    )
    if any(_PERSONAL_PATH.search(value) for value in portability_fields):
        raise ValueError("El candidato contiene una ruta personal y no puede exportarse")
    portable_rule_ids = tuple(
        rule.rule_id for rule in result.applied_rules if rule.scope == "builtin"
    )
    return PronunciationCorpusCase.from_dict(
        {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "case_id": _normalized(case_id),
            "status": "candidate",
            "language": result.language.split("-", 1)[0],
            "profile": result.profile.math_style,
            "category": _normalized(category),
            "written_text": result.written_text,
            "expected_spoken_text": result.spoken_text,
            "assertion_mode": "exact",
            "expected_warning_codes": [warning.code for warning in result.warnings],
            "expected_unsupported_fragments": list(result.unsupported_fragments),
            "semantic_anchors": [],
            "forbidden_fragments": [],
            "applied_rule_ids": list(portable_rule_ids),
            "tags": list(normalized_tags),
            "notes": normalized_notes,
            "source_kind": "ui_export",
            "source_reference": "",
            "created_at": "",
            "updated_at": "",
        }
    )


def export_corpus_candidate_json(candidate: PronunciationCorpusCase) -> str:
    if candidate.status != "candidate" or candidate.source_kind != "ui_export":
        raise ValueError("Sólo se exportan candidatos creados por la UI")
    candidate.validate()
    return deterministic_json(candidate.to_dict())
