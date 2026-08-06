from __future__ import annotations

import unicodedata

import pytest

from dialogue_studio.pronunciation.corpus import (
    CORPUS_SCHEMA_VERSION,
    PronunciationCorpusCase,
    PronunciationCorpusManifest,
)


def case_data(**changes: object) -> dict[str, object]:
    data: dict[str, object] = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "case_id": "es-calculus-derivative-001",
        "status": "approved",
        "language": "es",
        "profile": "classroom",
        "category": "calculus",
        "written_text": r"$\frac{dy}{dx}$",
        "expected_spoken_text": "de ye dividido entre de equis",
        "assertion_mode": "exact",
        "expected_warning_codes": [],
        "expected_unsupported_fragments": [],
        "semantic_anchors": [],
        "forbidden_fragments": ["pipeline"],
        "applied_rule_ids": [],
        "tags": ["derivative"],
        "notes": "Decisión pedagógica revisada.",
        "source_kind": "curated",
        "source_reference": "reference:calculus",
        "created_at": "2026-08-05T00:00:00+00:00",
        "updated_at": "2026-08-05T00:00:00+00:00",
    }
    data.update(changes)
    return data


def test_case_round_trip_and_profile() -> None:
    case = PronunciationCorpusCase.from_dict(case_data())
    assert case.to_dict() == case_data()
    assert case.pronunciation_profile().language == "es"
    assert case.pronunciation_profile().math_style == "classroom"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": 99}, "schema_version"),
        ({"case_id": "1"}, "case_id"),
        ({"status": "invented"}, "Estado"),
        ({"language": "fr"}, "Idioma"),
        ({"profile": "poetic"}, "Perfil"),
        ({"category": "closed category"}, "categoría"),
        ({"assertion_mode": "maybe"}, "Modo"),
        ({"expected_spoken_text": ""}, "expected_spoken_text"),
        (
            {"assertion_mode": "semantic", "semantic_anchors": []},
            "semantic_anchors",
        ),
        (
            {"assertion_mode": "warning_only", "expected_warning_codes": []},
            "warnings",
        ),
        ({"source_reference": "/home/user/private"}, "rutas personales"),
    ],
)
def test_case_validation_rejects_invalid_contract(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PronunciationCorpusCase.from_dict(case_data(**changes))


def test_candidate_may_capture_unapproved_current_output() -> None:
    candidate = PronunciationCorpusCase.from_dict(
        case_data(
            status="candidate",
            expected_spoken_text="",
            assertion_mode="semantic",
            semantic_anchors=[],
        )
    )
    assert candidate.status == "candidate"


def test_unicode_must_be_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "ecuación")
    with pytest.raises(ValueError, match="NFC"):
        PronunciationCorpusCase.from_dict(case_data(notes=decomposed))


def test_manifest_validation_and_deterministic_timestamp() -> None:
    manifest = PronunciationCorpusManifest.from_dict(
        {
            "schema_version": 1,
            "corpus_version": "1.0.0",
            "supported_languages": ["es", "en"],
            "default_profiles": {"es": "classroom", "en": "classroom"},
            "categories": ["calculus", "future_discipline"],
            "case_counts": {
                "total": 3,
                "approved": 1,
                "candidate": 1,
                "deprecated": 1,
            },
            "last_validated_at": "2026-08-05T00:00:00+00:00",
        }
    )
    assert manifest.categories[-1] == "future_discipline"
    assert manifest.to_dict()["last_validated_at"] == "2026-08-05T00:00:00+00:00"
