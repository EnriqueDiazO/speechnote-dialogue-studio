"""Application-level dialogue editing operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audio import AudioInfo, concatenate_waves, normalize_audio, probe_audio, sha256_file
from .models import (
    RECOVERABLE_SYNTHESIS_MESSAGE,
    DialogueProject,
    SpeakerProfile,
    SpeakerTTSConfig,
    Utterance,
    UtteranceTTSOverride,
    effective_tts_config,
    utc_now,
)
from .paths import safe_write_path
from .pronunciation import (
    PronunciationEngine,
    PronunciationProfile,
    PronunciationResult,
    PronunciationRule,
    PronunciationWarning,
)
from .pronunciation.import_export import record_rule_usage
from .qwen_client import synthesize_qwen_text
from .qwen_service import DEFAULT_GENERATION_OPTIONS
from .speechnote import synthesize_text
from .synthesis import SynthesisBusyError


@dataclass(frozen=True)
class GenerationPaths:
    raw: Path
    normalized: Path


_UNSET = object()


class PronunciationTransformationError(RuntimeError):
    """A recoverable preprocessing failure requiring an explicit written-text fallback."""


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
    speaker.tts = SpeakerTTSConfig(
        provider="speechnote",
        voice_id=model_id,
        voice_label=model_label,
        language="auto",
    )
    for utterance in project.utterances:
        if utterance.speaker_id == speaker_id:
            utterance.mark_stale()
    project.touch()


def update_speaker_tts(
    project: DialogueProject,
    speaker_id: str,
    config: SpeakerTTSConfig,
) -> None:
    config.validate()
    speaker = project.speaker(speaker_id)
    if speaker.tts_config == config:
        return
    speaker.tts = deepcopy(config)
    # Keep legacy fields useful to older readers without treating them as provider state.
    speaker.model_id = config.voice_id
    speaker.model_label = config.voice_label
    for utterance in project.utterances:
        if utterance.speaker_id == speaker_id:
            utterance.mark_stale()
    project.touch()


def update_speaker_name(project: DialogueProject, speaker_id: str, name: str) -> None:
    clean = name.strip()
    if not clean:
        raise ValueError("Cada hablante necesita un nombre")
    speaker = project.speaker(speaker_id)
    if speaker.name == clean:
        return
    speaker.name = clean
    for utterance in project.utterances:
        if utterance.speaker_id == speaker_id:
            utterance.mark_stale()
    project.touch()


def update_utterance_tts_override(
    project: DialogueProject,
    utterance_id: str,
    override: UtteranceTTSOverride | None,
) -> None:
    if override is not None:
        override.validate()
        if override.is_empty:
            override = None
    utterance = next(item for item in project.utterances if item.utterance_id == utterance_id)
    if utterance.tts_override == override:
        return
    utterance.tts_override = deepcopy(override)
    utterance.mark_stale()
    project.touch()


def effective_pronunciation_result(
    project: DialogueProject,
    utterance: Utterance,
    *,
    global_rules: list[PronunciationRule] | tuple[PronunciationRule, ...] = (),
    engine: PronunciationEngine | None = None,
) -> PronunciationResult:
    profile = project.pronunciation_profile
    if not utterance.use_pronunciation_engine:
        profile = replace(profile, enabled=False)
    processor = engine or PronunciationEngine()
    return processor.transform(
        utterance.text,
        profile=profile,
        rules=[*global_rules, *project.pronunciation_rules, *utterance.utterance_rules],
        manual_override=utterance.manual_spoken_text_override,
    )


def persist_pronunciation_result(
    utterance: Utterance,
    result: PronunciationResult,
) -> None:
    utterance.spoken_text = result.spoken_text
    utterance.written_text_hash = result.source_hash
    utterance.spoken_text_hash = hashlib.sha256(
        result.spoken_text.encode("utf-8")
    ).hexdigest()
    utterance.pronunciation_rules_hash = result.rules_hash
    utterance.pronunciation_engine_version = result.engine_version
    utterance.applied_pronunciation_rule_ids = [
        item.rule_id for item in result.applied_rules
    ]
    utterance.pronunciation_warnings = [asdict(item) for item in result.warnings]


def _pronunciation_effect_signature(result: PronunciationResult) -> tuple[object, ...]:
    return (
        result.spoken_text,
        tuple(item.rule_id for item in result.applied_rules),
        tuple((item.code, item.fragment) for item in result.warnings),
        result.unsupported_fragments,
    )


def update_project_pronunciation_rules(
    project: DialogueProject,
    rules: list[PronunciationRule],
    *,
    global_rules: list[PronunciationRule] | tuple[PronunciationRule, ...] = (),
) -> list[str]:
    for rule in rules:
        rule.validate()
        if rule.scope != "project":
            raise ValueError("Las reglas del proyecto necesitan alcance project")
    before = {
        utterance.utterance_id: _pronunciation_effect_signature(
            effective_pronunciation_result(project, utterance, global_rules=global_rules)
        )
        for utterance in project.utterances
    }
    project.pronunciation_rules = deepcopy(rules)
    affected: list[str] = []
    for utterance in project.utterances:
        after = _pronunciation_effect_signature(
            effective_pronunciation_result(project, utterance, global_rules=global_rules)
        )
        if before[utterance.utterance_id] != after:
            utterance.mark_stale()
            affected.append(utterance.utterance_id)
    project.touch()
    return affected


def mark_global_pronunciation_change(
    project: DialogueProject,
    *,
    old_rules: list[PronunciationRule] | tuple[PronunciationRule, ...],
    new_rules: list[PronunciationRule] | tuple[PronunciationRule, ...],
) -> list[str]:
    affected: list[str] = []
    for utterance in project.utterances:
        before = effective_pronunciation_result(project, utterance, global_rules=old_rules)
        after = effective_pronunciation_result(project, utterance, global_rules=new_rules)
        if _pronunciation_effect_signature(before) != _pronunciation_effect_signature(after):
            utterance.mark_stale()
            affected.append(utterance.utterance_id)
    if affected:
        project.touch()
    return affected


def update_pronunciation_profile(
    project: DialogueProject,
    profile: PronunciationProfile,
) -> None:
    profile.validate()
    if project.pronunciation_profile == profile:
        return
    project.pronunciation_profile = profile
    for utterance in project.utterances:
        utterance.mark_stale()
    project.touch()


def update_utterance_pronunciation(
    project: DialogueProject,
    utterance_id: str,
    *,
    enabled: bool | None = None,
    manual_override: str | None | object = _UNSET,
    rules: list[PronunciationRule] | None = None,
    global_rules: list[PronunciationRule] | tuple[PronunciationRule, ...] = (),
) -> None:
    utterance = next(item for item in project.utterances if item.utterance_id == utterance_id)
    before = _pronunciation_effect_signature(
        effective_pronunciation_result(project, utterance, global_rules=global_rules)
    )
    if enabled is not None:
        utterance.use_pronunciation_engine = enabled
    if manual_override is not _UNSET:
        utterance.manual_spoken_text_override = (
            str(manual_override).strip() if manual_override else None
        )
    if rules is not None:
        for rule in rules:
            rule.validate()
            if rule.scope != "utterance":
                raise ValueError("Las reglas de intervención necesitan alcance utterance")
        utterance.utterance_rules = deepcopy(rules)
    after = _pronunciation_effect_signature(
        effective_pronunciation_result(project, utterance, global_rules=global_rules)
    )
    if before != after:
        utterance.mark_stale()
    project.touch()


def audio_input_fingerprint(
    project: DialogueProject,
    utterance: Utterance,
    *,
    global_rules: list[PronunciationRule] | tuple[PronunciationRule, ...] = (),
    pronunciation_engine: PronunciationEngine | None = None,
    pronunciation_result: PronunciationResult | None = None,
) -> str:
    speaker = project.speaker(utterance.speaker_id)
    config = effective_tts_config(project, utterance)
    provider_model = (
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        if config.provider == "qwen"
        else "net.mkiol.SpeechNote"
    )
    generation_options = dict(config.generation_options)
    if config.provider == "qwen":
        generation_options = {**DEFAULT_GENERATION_OPTIONS, **generation_options}
    pronunciation = pronunciation_result or effective_pronunciation_result(
        project,
        utterance,
        global_rules=global_rules,
        engine=pronunciation_engine,
    )
    payload = {
        "written_text": utterance.text,
        "spoken_text": pronunciation.spoken_text,
        "pronunciation_profile": pronunciation.profile.to_dict(),
        "pronunciation_rules_hash": pronunciation.rules_hash,
        "manual_spoken_text_override": utterance.manual_spoken_text_override,
        "speaker_id": speaker.speaker_id,
        "speaker_name": speaker.name,
        "provider": config.provider,
        "provider_model": provider_model,
        "voice_id": config.voice_id,
        "language": config.language,
        "instruction_text": config.instruction_text,
        "generation_options": generation_options,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def add_speaker(
    project: DialogueProject,
    name: str,
    model_id: str = "",
    model_label: str = "",
    color_key: str = "accent",
    *,
    tts: SpeakerTTSConfig | None = None,
) -> SpeakerProfile:
    speaker = SpeakerProfile.create(name, model_id, model_label, color_key, tts=tts)
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
    duplicate.audio_fingerprint = None
    duplicate.spoken_text = None
    duplicate.written_text_hash = None
    duplicate.spoken_text_hash = None
    duplicate.pronunciation_rules_hash = None
    duplicate.pronunciation_engine_version = None
    duplicate.applied_pronunciation_rule_ids = []
    duplicate.pronunciation_warnings = []
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
    qwen_synthesizer: Callable[..., object] = synthesize_qwen_text,
    normalizer: Callable[..., AudioInfo] = normalize_audio,
    output_paths: GenerationPaths | None = None,
    global_rules: list[PronunciationRule] | tuple[PronunciationRule, ...] = (),
    pronunciation_engine: PronunciationEngine | None = None,
    allow_pronunciation_fallback: bool = False,
) -> Utterance:
    utterance = next(item for item in project.utterances if item.utterance_id == utterance_id)
    speaker = project.speaker(utterance.speaker_id)
    tts = effective_tts_config(project, utterance)
    if not utterance.text.strip():
        raise ValueError("No se puede sintetizar una intervención vacía")
    if not tts.voice_id.strip():
        raise ValueError(f"{speaker.name} no tiene una voz asignada")
    try:
        pronunciation = effective_pronunciation_result(
            project,
            utterance,
            global_rules=global_rules,
            engine=pronunciation_engine,
        )
    except Exception as exc:
        if not allow_pronunciation_fallback:
            raise PronunciationTransformationError(
                "No se pudo transformar la pronunciación. Revisa las reglas o activa "
                "explícitamente el fallback al texto escrito."
            ) from exc
        fallback_profile = replace(project.pronunciation_profile, enabled=False)
        pronunciation = PronunciationEngine().transform(
            utterance.text,
            profile=fallback_profile,
        )
        pronunciation = replace(
            pronunciation,
            warnings=(
                *pronunciation.warnings,
                PronunciationWarning(
                    code="explicit_written_text_fallback",
                    message=(
                        "Falló la transformación y se usó el texto escrito por "
                        "confirmación explícita."
                    ),
                ),
            ),
        )
    effective_text = pronunciation.spoken_text
    if not effective_text.strip():
        raise PronunciationTransformationError("El texto hablado resultante está vacío")
    paths = output_paths or prepare_generation_paths(project_dir, utterance, controlled_root)
    previous_state = (
        utterance.status,
        utterance.audio_relative_path,
        utterance.duration_seconds,
        utterance.sha256,
        utterance.error_message,
    )
    utterance.status = "generating"
    utterance.error_message = None
    utterance.updated_at = utc_now()
    try:
        if tts.provider == "qwen":
            generation_options = {
                **DEFAULT_GENERATION_OPTIONS,
                **tts.generation_options,
            }
            qwen_synthesizer(
                tts.voice_id,
                effective_text,
                tts.language,
                generation_options,
                paths.raw,
            )
        else:
            synthesizer(
                tts.voice_id,
                effective_text,
                paths.raw,
                controlled_root,
                probe=probe_audio,
            )
        info = normalizer(paths.raw, paths.normalized)
    except SynthesisBusyError:
        (
            utterance.status,
            utterance.audio_relative_path,
            utterance.duration_seconds,
            utterance.sha256,
            utterance.error_message,
        ) = previous_state
        raise
    except (KeyboardInterrupt, SystemExit):
        utterance.status = "stale"
        utterance.error_message = RECOVERABLE_SYNTHESIS_MESSAGE
        utterance.updated_at = utc_now()
        project.touch()
        raise
    except Exception as exc:
        utterance.status = "error"
        utterance.error_message = str(exc)
        utterance.updated_at = utc_now()
        project.touch()
        raise
    utterance.audio_relative_path = paths.normalized.relative_to(project_dir).as_posix()
    utterance.duration_seconds = info.duration_seconds
    utterance.sha256 = sha256_file(paths.normalized)
    persist_pronunciation_result(utterance, pronunciation)
    utterance.audio_fingerprint = audio_input_fingerprint(
        project,
        utterance,
        global_rules=global_rules,
        pronunciation_engine=pronunciation_engine,
        pronunciation_result=pronunciation,
    )
    applied_ids = {item.rule_id for item in pronunciation.applied_rules}
    project.pronunciation_rules = list(
        record_rule_usage(project.pronunciation_rules, applied_ids)
    )
    utterance.utterance_rules = list(
        record_rule_usage(utterance.utterance_rules, applied_ids)
    )
    if isinstance(global_rules, list):
        global_rules[:] = record_rule_usage(global_rules, applied_ids)
    utterance.status = "ready"
    utterance.error_message = None
    utterance.updated_at = utc_now()
    project.touch()
    return utterance


def prepare_generation_paths(
    project_dir: Path,
    utterance: Utterance,
    controlled_root: Path,
) -> GenerationPaths:
    token = uuid4().hex[:10]
    filename = f"{utterance.order:03d}-{utterance.utterance_id}-{token}.wav"
    raw = project_dir / "audio" / "raw" / filename
    normalized = project_dir / "audio" / "normalized" / filename
    return GenerationPaths(
        raw=safe_write_path(controlled_root, raw.relative_to(controlled_root)),
        normalized=safe_write_path(controlled_root, normalized.relative_to(controlled_root)),
    )


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
