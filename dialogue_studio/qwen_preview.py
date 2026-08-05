"""Safe, fingerprinted Qwen voice previews that never alter project audio."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .audio import AudioInfo, probe_audio
from .paths import AppPaths, safe_write_path
from .qwen_client import QwenClient, QwenClientError
from .qwen_service import DEFAULT_MODEL
from .synthesis import SynthesisBusyError, SynthesisCoordinator, run_with_synthesis_state


@dataclass(frozen=True)
class QwenPreview:
    fingerprint: str
    voice_id: str
    language: str
    path: Path | None
    duration_seconds: float | None
    elapsed_seconds: float | None
    cached: bool
    error: str | None = None


def preview_fingerprint(
    text: str,
    voice_id: str,
    language: str,
    generation_options: dict[str, int | float],
    *,
    model: str = DEFAULT_MODEL,
) -> str:
    payload = {
        "model": model,
        "text": text,
        "voice_id": voice_id,
        "language": language,
        "generation_options": generation_options,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def preview_directory(paths: AppPaths) -> Path:
    return safe_write_path(paths.root, "temporary/qwen-previews")


def _preview_path(paths: AppPaths, fingerprint: str, voice_id: str) -> Path:
    safe_voice = "".join(
        character
        for character in voice_id.lower()
        if character.isalnum() or character == "_"
    )
    if not safe_voice:
        raise ValueError("ID de voz Qwen inválido")
    return safe_write_path(
        paths.root,
        f"temporary/qwen-previews/{fingerprint[:24]}-{safe_voice}.wav",
    )


def _valid_preview(path: Path) -> AudioInfo | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return None
        info = probe_audio(path)
    except (OSError, RuntimeError, ValueError):
        return None
    if info.sample_rate != 24_000 or info.channels != 1 or info.duration_seconds <= 0:
        return None
    return info


def _preserve_invalid_preview(path: Path) -> Path | None:
    if not path.exists() or path.is_symlink():
        return None
    preserved = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    os.replace(path, preserved)
    return preserved


def generate_qwen_previews(
    *,
    paths: AppPaths,
    text: str,
    voice_ids: list[str],
    language: str,
    generation_options: dict[str, int | float],
    session_token: str,
    coordinator: SynthesisCoordinator,
    client: QwenClient,
) -> list[QwenPreview]:
    if not text.strip():
        raise ValueError("Escribe un texto para comparar voces")
    if not voice_ids:
        raise ValueError("Selecciona al menos una voz Qwen")
    directory = preview_directory(paths)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    results: list[QwenPreview] = []
    for voice_id in voice_ids:
        fingerprint = preview_fingerprint(text, voice_id, language, generation_options)
        destination = _preview_path(paths, fingerprint, voice_id)
        cached_info = _valid_preview(destination)
        if cached_info is not None:
            results.append(
                QwenPreview(
                    fingerprint,
                    voice_id,
                    language,
                    destination,
                    cached_info.duration_seconds,
                    0.0,
                    True,
                )
            )
            continue
        _preserve_invalid_preview(destination)
        try:
            response = run_with_synthesis_state(
                coordinator,
                f"qwen-preview-{voice_id}",
                destination,
                session_token,
                lambda current_voice=voice_id, current_path=destination: client.synthesize(
                    text=text.strip(),
                    speaker=current_voice,
                    language=language,
                    generation_options=generation_options,
                    output_path=current_path,
                ),
            )
            if not isinstance(response, dict):
                raise QwenClientError("El backend Qwen devolvió una respuesta inválida")
            info = _valid_preview(destination)
            if info is None:
                _preserve_invalid_preview(destination)
                raise QwenClientError("El preview Qwen no tiene el formato esperado")
            results.append(
                QwenPreview(
                    fingerprint,
                    voice_id,
                    language,
                    destination,
                    info.duration_seconds,
                    float(response.get("elapsed_seconds", 0.0)),
                    False,
                )
            )
        except (OSError, RuntimeError, ValueError, QwenClientError, SynthesisBusyError) as exc:
            results.append(
                QwenPreview(
                    fingerprint,
                    voice_id,
                    language,
                    None,
                    None,
                    None,
                    False,
                    str(exc),
                )
            )
    return results


def clear_qwen_previews(paths: AppPaths) -> int:
    directory = preview_directory(paths)
    if not directory.is_dir() or directory.is_symlink():
        return 0
    removed = 0
    for path in directory.glob("*.wav"):
        if (
            path.is_file()
            and not path.is_symlink()
            and path.parent.resolve() == directory.resolve()
        ):
            path.unlink()
            removed += 1
    return removed
