"""Safe application paths rooted in the user's XDG Music directory."""

from __future__ import annotations

import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def resolve_music_dir() -> Path:
    result = subprocess.run(
        ["xdg-user-dir", "MUSIC"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        raise RuntimeError("No se pudo resolver la carpeta Música con xdg-user-dir")
    return Path(value).expanduser().resolve()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    return slug[:48] or "dialogo"


def _assert_inside(root: Path, candidate: Path) -> Path:
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve(strict=False)
    if candidate_resolved != root_resolved and root_resolved not in candidate_resolved.parents:
        raise ValueError("La ruta sale de la carpeta controlada por la aplicación")
    return candidate_resolved


def safe_write_path(root: Path, relative: str | Path) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("La ruta de destino debe ser relativa y segura")
    lexical_root = root.absolute()
    lexical_target = lexical_root / relative_path
    cursor = lexical_target
    while cursor != lexical_root:
        if cursor.is_symlink():
            raise ValueError("No se permiten enlaces simbólicos como destino")
        cursor = cursor.parent
    return _assert_inside(root, lexical_target)


@dataclass(frozen=True)
class AppPaths:
    music_dir: Path

    @classmethod
    def discover(cls) -> AppPaths:
        return cls(resolve_music_dir())

    @property
    def root(self) -> Path:
        return self.music_dir / "SpeechNote Dialogue Studio"

    @property
    def projects(self) -> Path:
        return self.root / "projects"

    @property
    def temporary(self) -> Path:
        return self.root / "temporary"

    def ensure(self) -> None:
        for directory in (self.root, self.projects, self.temporary):
            if directory.is_symlink():
                raise ValueError(f"No se permite un enlace simbólico: {directory}")
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    def new_project_dir(self, title: str, project_id: str) -> Path:
        return safe_write_path(self.projects, f"{slugify(title)}-{project_id[:8]}")
