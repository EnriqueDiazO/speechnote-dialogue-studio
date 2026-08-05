"""Application-level dialogue editing operations."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audio import AudioInfo, concatenate_waves, normalize_audio, probe_audio, sha256_file
from .models import DialogueProject, SpeakerProfile, Utterance, utc_now
from .paths import safe_write_path
from .speechnote import synthesize_text


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


def generate_utterance(
    project: DialogueProject,
    project_dir: Path,
    utterance_id: str,
    controlled_root: Path,
    *,
    synthesizer: Callable[..., None] = synthesize_text,
    normalizer: Callable[..., AudioInfo] = normalize_audio,
) -> Utterance:
    utterance = next(item for item in project.utterances if item.utterance_id == utterance_id)
    speaker = project.speaker(utterance.speaker_id)
    if not utterance.text.strip():
        raise ValueError("No se puede sintetizar una intervención vacía")
    if not speaker.model_id.strip():
        raise ValueError(f"{speaker.name} no tiene una voz asignada")
    token = uuid4().hex[:10]
    filename = f"{utterance.order:03d}-{utterance.utterance_id}-{token}.wav"
    raw = project_dir / "audio" / "raw" / filename
    normalized = (
        project_dir
        / "audio"
        / "normalized"
        / f"{utterance.order:03d}-{utterance.utterance_id}-{token}.wav"
    )
    raw = safe_write_path(controlled_root, raw.relative_to(controlled_root))
    normalized = safe_write_path(controlled_root, normalized.relative_to(controlled_root))
    utterance.status = "generating"
    utterance.error_message = None
    utterance.updated_at = utc_now()
    try:
        synthesizer(
            speaker.model_id,
            utterance.text,
            raw,
            controlled_root,
            probe=probe_audio,
        )
        info = normalizer(raw, normalized)
    except Exception as exc:
        utterance.status = "error"
        utterance.error_message = str(exc)
        utterance.updated_at = utc_now()
        project.touch()
        raise
    utterance.audio_relative_path = normalized.relative_to(project_dir).as_posix()
    utterance.duration_seconds = info.duration_seconds
    utterance.sha256 = sha256_file(normalized)
    utterance.status = "ready"
    utterance.error_message = None
    utterance.updated_at = utc_now()
    project.touch()
    return utterance


def build_master(project: DialogueProject, project_dir: Path) -> tuple[Path, AudioInfo]:
    project.validate(require_utterance=True)
    unavailable = [item.order for item in project.utterances if item.status != "ready"]
    if unavailable:
        numbers = ", ".join(str(number) for number in unavailable)
        raise ValueError(f"Genera primero las intervenciones pendientes: {numbers}")
    paths: list[Path] = []
    for utterance in project.utterances:
        if not utterance.audio_relative_path:
            raise ValueError(f"La intervención {utterance.order} no tiene audio")
        path = safe_write_path(project_dir, utterance.audio_relative_path)
        if not path.is_file():
            raise ValueError(f"Ruta de audio no segura en la intervención {utterance.order}")
        paths.append(path)
    token = uuid4().hex[:10]
    destination = safe_write_path(project_dir, f"exports/dialogue-{token}.wav")
    return destination, concatenate_waves(paths, destination, project.pause_ms)


def project_metrics(project: DialogueProject) -> dict[str, Any]:
    generated = sum(item.status == "ready" for item in project.utterances)
    duration = sum(item.duration_seconds or 0 for item in project.utterances)
    if generated > 1:
        duration += (generated - 1) * project.pause_ms / 1000
    return {
        "utterances": len(project.utterances),
        "generated": generated,
        "pending": len(project.utterances) - generated,
        "duration_seconds": duration,
    }
