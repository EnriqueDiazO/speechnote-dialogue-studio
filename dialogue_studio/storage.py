"""Atomic JSON persistence for portable dialogue projects."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .models import RECOVERABLE_SYNTHESIS_MESSAGE, DialogueProject
from .paths import AppPaths, safe_write_path


def deterministic_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def persistent_project_data(project: DialogueProject) -> dict[str, object]:
    """Serialize transient generating states as recoverable stale states."""
    data = project.to_dict()
    for utterance in data["utterances"]:  # type: ignore[index]
        if utterance["status"] == "generating":
            utterance["status"] = "stale"
            utterance["error_message"] = RECOVERABLE_SYNTHESIS_MESSAGE
    return data


def atomic_write_text(path: Path, content: str) -> None:
    if path.is_symlink():
        raise ValueError("No se sobrescriben enlaces simbólicos")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ProjectRecord:
    project_id: str
    title: str
    directory: Path
    updated_at: str


class ProjectStore:
    def __init__(self, paths: AppPaths):
        self.paths = paths
        self.paths.ensure()

    def create_directory(self, project: DialogueProject) -> Path:
        directory = self.paths.new_project_dir(project.title, project.project_id)
        if directory.exists():
            raise FileExistsError("Ya existe una carpeta para este proyecto")
        directory.mkdir(mode=0o700, parents=False)
        for relative in ("audio/raw", "audio/normalized", "exports"):
            safe_write_path(directory, relative).mkdir(mode=0o700, parents=True)
        return directory

    def save(
        self,
        project: DialogueProject,
        directory: Path | None = None,
        *,
        allow_overwrite: bool = True,
    ) -> Path:
        project.normalize_order()
        project.validate()
        directory = directory or self.create_directory(project)
        directory = safe_write_path(self.paths.projects, directory.relative_to(self.paths.projects))
        target = safe_write_path(directory, "project.json")
        if target.exists() and not allow_overwrite:
            raise FileExistsError("El proyecto ya existe; confirma antes de sobrescribirlo")
        atomic_write_text(target, deterministic_json(persistent_project_data(project)))
        return directory

    def load(self, directory: Path) -> DialogueProject:
        directory = safe_write_path(self.paths.projects, directory.relative_to(self.paths.projects))
        target = safe_write_path(directory, "project.json")
        if target.is_symlink() or not target.is_file():
            raise FileNotFoundError("No se encontró project.json")
        data = json.loads(target.read_text(encoding="utf-8"))
        return DialogueProject.from_dict(data)

    def list_projects(self) -> list[ProjectRecord]:
        records: list[ProjectRecord] = []
        if not self.paths.projects.exists():
            return records
        for project_file in sorted(self.paths.projects.glob("*/project.json")):
            if project_file.is_symlink() or project_file.parent.is_symlink():
                continue
            try:
                project = DialogueProject.from_dict(
                    json.loads(project_file.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            records.append(
                ProjectRecord(
                    project.project_id,
                    project.title,
                    project_file.parent,
                    project.updated_at,
                )
            )
        return sorted(records, key=lambda item: item.updated_at, reverse=True)
