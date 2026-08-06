from __future__ import annotations

import json

import pytest

from dialogue_studio.pronunciation import PronunciationEngine, PronunciationProfile
from dialogue_studio.pronunciation.corpus_export import (
    build_corpus_candidate,
    export_corpus_candidate_json,
)


def test_ui_candidate_is_portable_valid_and_deterministic() -> None:
    result = PronunciationEngine().transform(
        r"La pérdida es $L(\theta)$.",
        profile=PronunciationProfile(language="es", math_style="classroom"),
    )
    candidate = build_corpus_candidate(
        result,
        case_id="es-machine-learning-ui-001",
        category="machine_learning",
        tags=["curso", "loss", "curso"],
        notes="Lectura para revisión.",
    )
    exported = export_corpus_candidate_json(candidate)
    data = json.loads(exported)
    assert data["status"] == "candidate"
    assert data["written_text"] == result.written_text
    assert data["expected_spoken_text"] == result.spoken_text
    assert data["expected_warning_codes"] == []
    assert data["expected_unsupported_fragments"] == []
    assert data["applied_rule_ids"] == ["builtin-math-classroom"]
    assert data["tags"] == ["curso", "loss"]
    assert data["source_kind"] == "ui_export"
    assert exported == export_corpus_candidate_json(candidate)
    assert "project_id" not in exported
    assert "utterance_id" not in exported
    assert "audio" not in exported


def test_ui_candidate_rejects_personal_paths() -> None:
    result = PronunciationEngine().transform(
        "/home/private-user/secret.txt",
        profile=PronunciationProfile(language="en"),
    )
    with pytest.raises(ValueError, match="ruta personal"):
        build_corpus_candidate(
            result,
            case_id="en-edge-cases-private-path-001",
            category="edge_cases",
        )
