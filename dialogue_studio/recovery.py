"""Inspection and safe recovery of interrupted synthesis state."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from .audio import AudioInfo, probe_audio, sha256_file
from .models import (
    RECOVERABLE_SYNTHESIS_MESSAGE,
    DialogueProject,
    Utterance,
    utc_now,
)
from .paths import safe_write_path

RECOVERABLE_MESSAGE = RECOVERABLE_SYNTHESIS_MESSAGE
LEGACY_BUSY_MESSAGE = "ya hay una síntesis en curso"


@dataclass(frozen=True)
class RecoveryItem:
    utterance_id: str
    order: int
    previous_status: str
    audio_state: str
    proposed_action: str
    audio_relative_path: str | None
    duration_seconds: float | None = None
    sha256: str | None = None


@dataclass
class RecoveryReport:
    items: list[RecoveryItem] = field(default_factory=list)
    changed: bool = False
    recovered_ready: int = 0
    converted_recoverable: int = 0
    preserved_partial: int = 0

    @property
    def affected_count(self) -> int:
        return len(self.items)


Probe = Callable[[Path], AudioInfo]


def _is_recovery_candidate(utterance: Utterance) -> bool:
    legacy_busy = (
        utterance.status == "error"
        and LEGACY_BUSY_MESSAGE in (utterance.error_message or "").lower()
    )
    return utterance.status == "generating" or legacy_busy


def _inspect_audio(
    utterance: Utterance,
    project_dir: Path,
    probe: Probe,
) -> tuple[str, str, float | None, str | None]:
    relative = utterance.audio_relative_path
    if not relative:
        return "missing", "mark_stale", None, None
    try:
        candidate = safe_write_path(project_dir, relative)
    except ValueError:
        return "invalid", "preserve_partial", None, None
    if not candidate.is_file():
        return "missing", "mark_stale", None, None
    try:
        with candidate.open("rb") as handle:
            header = handle.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("invalid RIFF/WAVE header")
        info = probe(candidate)
        if info.duration_seconds <= 0:
            raise ValueError("empty WAV")
        if utterance.utterance_id not in candidate.name:
            return "mismatched", "preserve_partial", None, None
        return "valid", "mark_ready", info.duration_seconds, sha256_file(candidate)
    except (OSError, RuntimeError, ValueError):
        return "invalid", "preserve_partial", None, None


def inspect_interrupted_synthesis(
    project: DialogueProject,
    project_dir: Path,
    *,
    probe: Probe = probe_audio,
) -> RecoveryReport:
    """Describe recovery work without changing the project or its files."""
    report = RecoveryReport()
    for utterance in project.utterances:
        if not _is_recovery_candidate(utterance):
            continue
        audio_state, action, duration, digest = _inspect_audio(utterance, project_dir, probe)
        report.items.append(
            RecoveryItem(
                utterance_id=utterance.utterance_id,
                order=utterance.order,
                previous_status=utterance.status,
                audio_state=audio_state,
                proposed_action=action,
                audio_relative_path=utterance.audio_relative_path,
                duration_seconds=duration,
                sha256=digest,
            )
        )
    return report


def _preserve_partial(project_dir: Path, relative: str | None) -> str | None:
    if not relative:
        return None
    try:
        source = safe_write_path(project_dir, relative)
    except ValueError:
        return None
    if not source.is_file() or source.is_symlink():
        return None
    recovery_dir = safe_write_path(project_dir, "audio/recovery")
    recovery_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = f"{source.name}.{uuid4().hex[:8]}.partial"
    destination = safe_write_path(project_dir, Path("audio/recovery") / name)
    os.replace(source, destination)
    return destination.relative_to(project_dir).as_posix()


def recover_interrupted_synthesis(
    project: DialogueProject,
    project_dir: Path,
    *,
    probe: Probe = probe_audio,
) -> RecoveryReport:
    """Normalize interrupted states without persisting or deleting audio."""
    report = inspect_interrupted_synthesis(project, project_dir, probe=probe)
    by_id = {utterance.utterance_id: utterance for utterance in project.utterances}
    recovered_items: list[RecoveryItem] = []
    for item in report.items:
        utterance = by_id[item.utterance_id]
        recovered_path: str | None = None
        if item.proposed_action == "mark_ready":
            utterance.status = "ready"
            utterance.duration_seconds = item.duration_seconds
            utterance.sha256 = item.sha256
            utterance.error_message = None
            report.recovered_ready += 1
        else:
            if item.proposed_action == "preserve_partial":
                recovered_path = _preserve_partial(project_dir, utterance.audio_relative_path)
                if recovered_path:
                    report.preserved_partial += 1
            utterance.status = "stale"
            utterance.audio_relative_path = None
            utterance.duration_seconds = None
            utterance.sha256 = None
            utterance.error_message = RECOVERABLE_MESSAGE
            report.converted_recoverable += 1
        utterance.updated_at = utc_now()
        recovered_items.append(
            RecoveryItem(
                utterance_id=item.utterance_id,
                order=item.order,
                previous_status=item.previous_status,
                audio_state=item.audio_state,
                proposed_action=item.proposed_action,
                audio_relative_path=recovered_path or item.audio_relative_path,
                duration_seconds=item.duration_seconds,
                sha256=item.sha256,
            )
        )
    if recovered_items:
        project.touch()
        report.changed = True
        report.items = recovered_items
    return report
