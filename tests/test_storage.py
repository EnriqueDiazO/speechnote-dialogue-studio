from __future__ import annotations

import json
from pathlib import Path

import pytest

from dialogue_studio.models import DialogueProject
from dialogue_studio.paths import AppPaths, safe_write_path, slugify
from dialogue_studio.storage import ProjectStore


def test_slug_is_safe_and_unicode_friendly() -> None:
    assert slugify("Álgebra y Redes") == "algebra-y-redes"
    assert slugify("***") == "dialogo"


def test_atomic_save_and_load_has_only_relative_audio_paths(tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "Música")
    store = ProjectStore(paths)
    project = DialogueProject.sample()
    utterance = project.utterances[0]
    utterance.audio_relative_path = f"audio/normalized/{utterance.utterance_id}.wav"
    directory = store.save(project)
    payload = (directory / "project.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in payload
    assert json.loads(payload)["utterances"][0]["audio_relative_path"].startswith("audio/")
    loaded = store.load(directory)
    assert loaded.to_dict() == project.to_dict()
    assert store.list_projects()[0].project_id == project.project_id


def test_save_refuses_unconfirmed_overwrite(tmp_path: Path) -> None:
    store = ProjectStore(AppPaths(tmp_path / "Music"))
    project = DialogueProject.new()
    directory = store.save(project)
    with pytest.raises(FileExistsError, match="confirma"):
        store.save(project, directory, allow_overwrite=False)


def test_generating_is_never_persisted_as_an_active_lock(tmp_path: Path) -> None:
    store = ProjectStore(AppPaths(tmp_path / "Music"))
    project = DialogueProject.new()
    project.utterances[0].status = "generating"

    directory = store.save(project)
    payload = json.loads((directory / "project.json").read_text(encoding="utf-8"))
    loaded = store.load(directory)

    assert project.utterances[0].status == "generating"
    assert payload["utterances"][0]["status"] == "stale"
    assert loaded.utterances[0].status == "stale"
    assert not any(key in payload for key in ("busy", "active_synthesis", "active_synthesis_id"))


def test_safe_write_path_rejects_escape_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="segura"):
        safe_write_path(root, "../escape")
    real = root / "real"
    real.mkdir()
    link = root / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("El sistema no admite symlinks")
    with pytest.raises(ValueError, match="simbólicos"):
        safe_write_path(root, "link/file.json")
