"""Application-level dialogue editing operations."""

from __future__ import annotations

from copy import deepcopy

from .models import DialogueProject, SpeakerProfile, Utterance, utc_now


def update_utterance(
    project: DialogueProject,
    utterance_id: str,
    *,
    text: str | None = None,
    speaker_id: str | None = None,
) -> None:
    utterance = next(item for item in project.utterances if item.utterance_id == utterance_id)
    changed = False
    if text is not None and text != utterance.text:
        utterance.text = text
        changed = True
    if speaker_id is not None and speaker_id != utterance.speaker_id:
        project.speaker(speaker_id)
        utterance.speaker_id = speaker_id
        changed = True
    if changed:
        utterance.mark_stale()
        project.touch()


def update_speaker_voice(
    project: DialogueProject, speaker_id: str, model_id: str, model_label: str
) -> None:
    speaker = project.speaker(speaker_id)
    if speaker.model_id == model_id and speaker.model_label == model_label:
        return
    speaker.model_id = model_id
    speaker.model_label = model_label
    for utterance in project.utterances:
        if utterance.speaker_id == speaker_id:
            utterance.mark_stale()
    project.touch()


def add_speaker(
    project: DialogueProject,
    name: str,
    model_id: str = "",
    model_label: str = "",
    color_key: str = "accent",
) -> SpeakerProfile:
    speaker = SpeakerProfile.create(name, model_id, model_label, color_key)
    project.speakers.append(speaker)
    project.touch()
    return speaker


def remove_speaker(
    project: DialogueProject, speaker_id: str, *, confirm_in_use: bool = False
) -> None:
    if len(project.speakers) == 1:
        raise ValueError("El proyecto necesita al menos un hablante")
    in_use = any(item.speaker_id == speaker_id for item in project.utterances)
    if in_use and not confirm_in_use:
        raise ValueError("El hablante está en uso; confirma antes de eliminarlo")
    if in_use:
        raise ValueError("Reasigna sus intervenciones antes de eliminar el hablante")
    project.speakers = [item for item in project.speakers if item.speaker_id != speaker_id]
    project.touch()


def add_utterance(
    project: DialogueProject, speaker_id: str | None = None, text: str = ""
) -> Utterance:
    selected = speaker_id or project.speakers[0].speaker_id
    project.speaker(selected)
    utterance = Utterance.create(len(project.utterances) + 1, selected, text)
    project.utterances.append(utterance)
    project.touch()
    return utterance


def move_utterance(project: DialogueProject, utterance_id: str, offset: int) -> None:
    index = next(
        i for i, item in enumerate(project.utterances) if item.utterance_id == utterance_id
    )
    destination = index + offset
    if destination < 0 or destination >= len(project.utterances):
        return
    project.utterances[index], project.utterances[destination] = (
        project.utterances[destination],
        project.utterances[index],
    )
    project.normalize_order()


def duplicate_utterance(project: DialogueProject, utterance_id: str) -> Utterance:
    index = next(
        i for i, item in enumerate(project.utterances) if item.utterance_id == utterance_id
    )
    source = project.utterances[index]
    duplicate = deepcopy(source)
    duplicate.utterance_id = Utterance.create(1, source.speaker_id).utterance_id
    duplicate.audio_relative_path = None
    duplicate.duration_seconds = None
    duplicate.sha256 = None
    duplicate.status = "draft"
    duplicate.error_message = None
    duplicate.created_at = utc_now()
    duplicate.updated_at = duplicate.created_at
    project.utterances.insert(index + 1, duplicate)
    project.normalize_order()
    return duplicate


def delete_utterance(project: DialogueProject, utterance_id: str) -> None:
    project.utterances = [item for item in project.utterances if item.utterance_id != utterance_id]
    project.normalize_order()
