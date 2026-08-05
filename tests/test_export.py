from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from dialogue_studio.audio import concatenate_waves, sha256_file
from dialogue_studio.export import export_project_zip
from dialogue_studio.models import DialogueProject
from dialogue_studio.storage import deterministic_json


def _ready_project(make_wav, directory: Path) -> tuple[DialogueProject, Path]:
    project = DialogueProject.sample()
    segments = []
    for utterance in project.utterances:
        relative = f"audio/normalized/{utterance.order:03d}-{utterance.utterance_id}.wav"
        path = make_wav(directory / relative, duration=0.1)
        segments.append(path)
        utterance.audio_relative_path = relative
        utterance.duration_seconds = 0.1
        utterance.sha256 = sha256_file(path)
        utterance.status = "ready"
    (directory / "exports").mkdir(parents=True)
    (directory / "project.json").write_text(deterministic_json(project.to_dict()), encoding="utf-8")
    master = directory / "exports" / "dialogue.wav"
    concatenate_waves(segments, master, project.pause_ms)
    return project, master


def test_portable_zip_manifest_scripts_and_deterministic_order(make_wav, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project, master = _ready_project(make_wav, project_dir)
    first = project_dir / "exports" / "first.zip"
    second = project_dir / "exports" / "second.zip"
    _, manifest = export_project_zip(project, project_dir, first, master_wav=master)
    export_project_zip(project, project_dir, second, master_wav=master)
    assert first.read_bytes() == second.read_bytes()
    assert manifest["format"] == "speechnote-dialogue-studio-project"
    assert all(not Path(item["path"]).is_absolute() for item in manifest["files"])
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "speech-dialogue-project/project.json" in names
        assert "speech-dialogue-project/manifest.json" in names
        assert "speech-dialogue-project/audio/dialogue.wav" in names
        assert "speech-dialogue-project/script/dialogue.txt" in names
        assert len([name for name in names if "/audio/segments/" in name]) == 3
        portable = json.loads(archive.read("speech-dialogue-project/project.json"))
        assert all(
            item["audio_relative_path"].startswith("audio/segments/")
            for item in portable["utterances"]
        )
        assert str(tmp_path) not in archive.read("speech-dialogue-project/manifest.json").decode()


def test_zip_refuses_overwrite_external_destination_and_symlink(make_wav, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project, master = _ready_project(make_wav, project_dir)
    destination = project_dir / "exports" / "project.zip"
    export_project_zip(project, project_dir, destination, master_wav=master)
    with pytest.raises(FileExistsError, match="confirma"):
        export_project_zip(project, project_dir, destination, master_wav=master)
    with pytest.raises(ValueError, match="dentro"):
        export_project_zip(project, project_dir, tmp_path / "outside.zip", master_wav=master)
    link = project_dir / "audio" / "normalized" / "linked.wav"
    try:
        link.symlink_to(project_dir / project.utterances[0].audio_relative_path)
    except OSError:
        pytest.skip("El sistema no admite symlinks")
    project.utterances[0].audio_relative_path = "audio/normalized/linked.wav"
    with pytest.raises(ValueError, match="simbólicos"):
        export_project_zip(
            project,
            project_dir,
            project_dir / "exports" / "linked.zip",
            master_wav=master,
        )
