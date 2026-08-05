from __future__ import annotations

from uuid import UUID

import pytest

from dialogue_studio.models import DialogueProject
from dialogue_studio.service import (
    add_speaker,
    add_utterance,
    delete_utterance,
    duplicate_utterance,
    move_utterance,
    remove_speaker,
    update_speaker_voice,
    update_utterance,
)
from dialogue_studio.storage import deterministic_json


def test_minimum_project_has_generic_valid_domain() -> None:
    project = DialogueProject.new()
    project.validate()
    UUID(project.project_id)
    assert len(project.speakers) == 2
    assert len(project.utterances) == 1
    assert project.language == "es-MX"
    assert project.pause_ms == 650
    assert project.utterances[0].order == 1
    assert project.utterances[0].status == "draft"


def test_sample_has_expected_speakers_and_script() -> None:
    project = DialogueProject.sample()
    assert [speaker.name for speaker in project.speakers] == ["Profesor", "Estudiante"]
    assert [speaker.model_id for speaker in project.speakers] == [
        "es_piper_mx_claude_high",
        "es_piper_es_sharvard_medium_1",
    ]
    assert len(project.utterances) == 3


def test_serialization_is_deterministic_and_tolerates_known_optional_fields() -> None:
    project = DialogueProject.sample()
    encoded = deterministic_json(project.to_dict())
    assert encoded == deterministic_json(project.to_dict())
    data = project.to_dict()
    del data["description"]
    del data["speakers"][0]["enabled"]
    loaded = DialogueProject.from_dict(data)
    assert loaded.description == ""
    assert loaded.speakers[0].enabled is True


def test_future_schema_is_rejected_clearly() -> None:
    data = DialogueProject.new().to_dict()
    data["schema_version"] = 99
    with pytest.raises(ValueError, match="admite hasta"):
        DialogueProject.from_dict(data)


@pytest.mark.parametrize("path", ["/home/user/audio.wav", "../audio.wav", "audio\\x.wav"])
def test_non_portable_audio_paths_are_rejected(path: str) -> None:
    project = DialogueProject.new()
    project.utterances[0].audio_relative_path = path
    with pytest.raises(ValueError, match="relativas"):
        project.validate()


def test_text_speaker_and_voice_changes_mark_ready_audio_stale() -> None:
    project = DialogueProject.new()
    utterance = project.utterances[0]
    utterance.status = "ready"
    utterance.audio_relative_path = f"audio/normalized/{utterance.utterance_id}.wav"
    update_utterance(project, utterance.utterance_id, text="Nuevo texto")
    assert utterance.status == "stale"
    utterance.status = "ready"
    update_utterance(project, utterance.utterance_id, speaker_id=project.speakers[1].speaker_id)
    assert utterance.status == "stale"
    utterance.status = "ready"
    update_speaker_voice(
        project,
        project.speakers[1].speaker_id,
        "new_model",
        "Nueva voz",
    )
    assert utterance.status == "stale"


def test_add_move_duplicate_delete_preserves_consecutive_order() -> None:
    project = DialogueProject.new()
    first = project.utterances[0]
    first.text = "Uno"
    second = add_utterance(project, text="Dos")
    move_utterance(project, second.utterance_id, -1)
    assert [item.text for item in project.utterances] == ["Dos", "Uno"]
    duplicate = duplicate_utterance(project, first.utterance_id)
    assert duplicate.utterance_id != first.utterance_id
    assert duplicate.status == "draft"
    delete_utterance(project, second.utterance_id)
    assert [item.order for item in project.utterances] == [1, 2]


def test_multiple_speakers_and_in_use_delete_guard() -> None:
    project = DialogueProject.new()
    guest = add_speaker(project, "Invitada", "model", "Model")
    add_utterance(project, guest.speaker_id, "Hola")
    with pytest.raises(ValueError, match="en uso"):
        remove_speaker(project, guest.speaker_id)
    project.utterances = [
        item for item in project.utterances if item.speaker_id != guest.speaker_id
    ]
    project.normalize_order()
    remove_speaker(project, guest.speaker_id)
    assert guest not in project.speakers
