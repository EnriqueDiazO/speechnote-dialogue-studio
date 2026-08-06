"""Portable project manifest and ZIP export."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from .audio import AudioError, export_mp3, probe_audio, probe_wave, sha256_file
from .models import DialogueProject
from .paths import safe_write_path, slugify
from .storage import atomic_write_text, deterministic_json

ZIP_ROOT = PurePosixPath("speech-dialogue-project")
INDIVIDUAL_EXPORT_ROOT = PurePosixPath("individual-exports")


def individual_export_filename(
    project_title: str,
    order: int,
    speaker_name: str,
    extension: str,
) -> str:
    """Return a predictable download name with no path-significant characters."""

    normalized_extension = extension.lower().lstrip(".")
    if normalized_extension not in {"wav", "mp3"}:
        raise ValueError("La exportación individual sólo admite WAV o MP3")
    if order < 1:
        raise ValueError("El número de intervención debe ser positivo")
    project_slug = slugify(project_title).replace("-", "_")
    speaker_slug = slugify(speaker_name).replace("-", "_")
    return (
        f"{project_slug}_intervencion_{order:02d}_{speaker_slug}."
        f"{normalized_extension}"
    )


def individual_export_widget_key(utterance_id: str, extension: str) -> str:
    """Build a stable widget key from the persistent utterance identity."""

    UUID(utterance_id)
    normalized_extension = extension.lower().lstrip(".")
    if normalized_extension not in {"wav", "mp3"}:
        raise ValueError("La exportación individual sólo admite WAV o MP3")
    return f"intervention-export-{normalized_extension}-{utterance_id}"


def individual_mp3_path(
    temporary_root: Path,
    utterance_id: str,
    source_sha256: str,
) -> Path:
    """Resolve the cached MP3 path without using order or user-provided names."""

    UUID(utterance_id)
    digest = source_sha256.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("La huella del WAV no es válida")
    relative = INDIVIDUAL_EXPORT_ROOT / utterance_id / f"{digest}.mp3"
    return safe_write_path(temporary_root, relative)


def is_mp3_file(path: Path) -> bool:
    """Perform a cheap container signature check for a cached MP3 artifact."""

    try:
        with path.open("rb") as handle:
            header = handle.read(3)
    except OSError:
        return False
    return header == b"ID3" or (
        len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
    )


def individual_wav_source(source_wav: Path) -> Path:
    """Validate and reuse one normalized intervention WAV without copying it."""

    info = probe_wave(source_wav)
    if not info.is_master_format:
        raise AudioError("El audio individual no tiene el formato WAV maestro esperado")
    return source_wav


def prepare_individual_mp3(
    source_wav: Path,
    temporary_root: Path,
    utterance_id: str,
) -> tuple[Path, bool]:
    """Create or reuse the MP3 for exactly one normalized intervention."""

    individual_wav_source(source_wav)
    destination = individual_mp3_path(
        temporary_root,
        utterance_id,
        sha256_file(source_wav),
    )
    if is_mp3_file(destination):
        return destination, True
    info = export_mp3(source_wav, destination)
    if info.codec != "mp3" or not is_mp3_file(destination):
        destination.unlink(missing_ok=True)
        raise AudioError("FFmpeg no produjo un archivo MP3 válido")
    return destination, False


def _safe_source(path: Path, project_dir: Path) -> Path:
    lexical_root = project_dir.absolute()
    cursor = path.absolute()
    while cursor != lexical_root:
        if cursor.is_symlink():
            raise ValueError("El export no admite enlaces simbólicos")
        if lexical_root not in cursor.parents:
            raise ValueError("El export sólo admite archivos del proyecto")
        cursor = cursor.parent
    resolved = path.resolve()
    root = project_dir.resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("El export sólo admite archivos regulares del proyecto")
    return resolved


def _file_entry(
    archive_path: str,
    role: str,
    *,
    data: bytes | None = None,
    source: Path | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    if (data is None) == (source is None):
        raise ValueError("La entrada necesita datos o archivo, pero no ambos")
    payload_size = len(data) if data is not None else source.stat().st_size  # type: ignore[union-attr]
    digest = (
        hashlib.sha256(data).hexdigest() if data is not None else sha256_file(source)  # type: ignore[arg-type]
    )
    media_type = mimetypes.guess_type(archive_path)[0] or "application/octet-stream"
    return {
        "path": archive_path,
        "role": role,
        "size": payload_size,
        "sha256": digest,
        "media_type": media_type,
        "duration_seconds": duration_seconds,
    }


def render_script_text(project: DialogueProject) -> str:
    lines: list[str] = []
    for utterance in project.utterances:
        speaker = project.speaker(utterance.speaker_id)
        lines.extend((f"{speaker.name.upper()}:", utterance.text.strip(), ""))
    return "\n".join(lines).rstrip() + "\n"


def render_script_markdown(project: DialogueProject) -> str:
    lines = [f"# {project.title}", ""]
    if project.description:
        lines.extend((project.description, ""))
    for utterance in project.utterances:
        speaker = project.speaker(utterance.speaker_id)
        lines.extend((f"## {speaker.name}", "", utterance.text.strip(), ""))
    return "\n".join(lines).rstrip() + "\n"


def render_export_readme(project: DialogueProject) -> str:
    voices = "\n".join(
        f"- {speaker.name}: {speaker.tts_config.voice_label or speaker.tts_config.voice_id} "
        f"({speaker.tts_config.provider})"
        for speaker in project.speakers
    )
    return (
        f"# {project.title}\n\n"
        "Proyecto portable de SpeechNote Dialogue Studio.\n\n"
        f"- Idioma: {project.language}\n"
        f"- Pausa entre intervenciones: {project.pause_ms} ms\n"
        f"- Intervenciones: {len(project.utterances)}\n"
        "- Audio maestro: WAV PCM 16-bit, 48000 Hz, mono\n\n"
        "## Voces\n\n"
        f"{voices}\n\n"
        "Los audios se generaron localmente mediante los proveedores configurados. "
        "El ZIP no contiene "
        "modelos de voz. Reproduce `audio/dialogue.wav` o, si existe, "
        "`audio/dialogue.mp3`.\n"
    )


def build_manifest(
    project: DialogueProject,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    ready_ids = [
        utterance.utterance_id
        for utterance in project.utterances
        if utterance.status == "ready" and utterance.audio_relative_path
    ]
    pending_ids = [
        utterance.utterance_id
        for utterance in project.utterances
        if utterance.utterance_id not in ready_ids
    ]
    project_data = project.to_dict()
    return {
        "format": "speechnote-dialogue-studio-project",
        "schema_version": project.schema_version,
        "project_id": project.project_id,
        "title": project.title,
        "language": project.language,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "speakers": project_data["speakers"],
        "utterances": project_data["utterances"],
        "ready_utterance_ids": ready_ids,
        "pending_utterance_ids": pending_ids,
        "pause_ms": project.pause_ms,
        "files": sorted(files, key=lambda item: item["path"]),
    }


def _zip_write_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Ruta ZIP insegura")
    info = zipfile.ZipInfo(name, date_time=(2024, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def export_project_zip(
    project: DialogueProject,
    project_dir: Path,
    destination: Path,
    *,
    master_wav: Path | None = None,
    master_mp3: Path | None = None,
    allow_overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    project.validate(require_utterance=True)
    root = project_dir.resolve()
    resolved_destination = destination.resolve(strict=False)
    if root not in resolved_destination.parents:
        raise ValueError("El ZIP debe guardarse dentro del proyecto")
    if destination.exists() and not allow_overwrite:
        raise FileExistsError("El ZIP ya existe; confirma antes de sobrescribirlo")
    if destination.is_symlink():
        raise ValueError("No se sobrescriben enlaces simbólicos")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    portable = deepcopy(project)
    payloads: dict[str, bytes] = {}
    sources: dict[str, Path] = {}
    entries: list[dict[str, Any]] = []
    for utterance in portable.utterances:
        if utterance.status != "ready" or not utterance.audio_relative_path:
            utterance.audio_relative_path = None
            utterance.duration_seconds = None
            utterance.sha256 = None
            continue
        try:
            source = _safe_source(project_dir / utterance.audio_relative_path, project_dir)
        except (OSError, ValueError):
            utterance.audio_relative_path = None
            utterance.duration_seconds = None
            utterance.sha256 = None
            utterance.status = "error"
            utterance.error_message = "El audio no estaba disponible al exportar"
            continue
        archive_relative = f"audio/segments/{utterance.order:03d}-{utterance.utterance_id}.wav"
        utterance.audio_relative_path = archive_relative
        sources[archive_relative] = source
        entries.append(
            _file_entry(
                archive_relative,
                "utterance",
                source=source,
                duration_seconds=utterance.duration_seconds,
            )
        )

    script_text = render_script_text(portable).encode("utf-8")
    script_markdown = render_script_markdown(portable).encode("utf-8")
    readme = render_export_readme(portable).encode("utf-8")
    payloads["script/dialogue.txt"] = script_text
    payloads["script/dialogue.md"] = script_markdown
    payloads["README.md"] = readme
    entries.extend(
        [
            _file_entry("script/dialogue.txt", "script", data=script_text),
            _file_entry("script/dialogue.md", "script", data=script_markdown),
            _file_entry("README.md", "documentation", data=readme),
        ]
    )
    if master_wav is not None:
        source = _safe_source(master_wav, project_dir)
        sources["audio/dialogue.wav"] = source
        entries.append(
            _file_entry(
                "audio/dialogue.wav",
                "master",
                source=source,
                duration_seconds=probe_audio(source).duration_seconds,
            )
        )
    if master_mp3 is not None:
        source = _safe_source(master_mp3, project_dir)
        sources["audio/dialogue.mp3"] = source
        entries.append(
            _file_entry(
                "audio/dialogue.mp3",
                "master-mp3",
                source=source,
                duration_seconds=probe_audio(source).duration_seconds,
            )
        )
    project_json = deterministic_json(portable.to_dict()).encode("utf-8")
    payloads["project.json"] = project_json
    entries.append(_file_entry("project.json", "project", data=project_json))
    manifest = build_manifest(portable, entries)
    manifest_json = deterministic_json(manifest).encode("utf-8")
    payloads["manifest.json"] = manifest_json
    atomic_write_text(project_dir / "manifest.json", manifest_json.decode("utf-8"))

    descriptor, name = tempfile.mkstemp(suffix=".zip", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for relative in sorted(payloads.keys() | sources.keys()):
                data = payloads.get(relative)
                if data is None:
                    data = sources[relative].read_bytes()
                _zip_write_bytes(archive, str(ZIP_ROOT / relative), data)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, manifest
