from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from dialogue_studio.audio import (
    AudioError,
    concatenate_waves,
    has_ffmpeg,
    probe_audio,
    sha256_file,
)
from dialogue_studio.export import (
    export_project_zip,
    individual_export_filename,
    individual_export_widget_key,
    individual_mp3_path,
    individual_wav_source,
    is_mp3_file,
    prepare_individual_mp3,
)
from dialogue_studio.models import DialogueProject
from dialogue_studio.service import duplicate_utterance, move_utterance
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
    project.utterances[0].spoken_text = "Texto hablado derivado"
    first = project_dir / "exports" / "first.zip"
    second = project_dir / "exports" / "second.zip"
    _, manifest = export_project_zip(project, project_dir, first, master_wav=master)
    export_project_zip(project, project_dir, second, master_wav=master)
    assert first.read_bytes() == second.read_bytes()
    assert manifest["format"] == "speechnote-dialogue-studio-project"
    assert manifest["pronunciation_profile"]["enabled"] is True
    assert manifest["pending_utterance_ids"] == []
    assert manifest["ready_utterance_ids"] == [
        utterance.utterance_id for utterance in project.utterances
    ]
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
        assert portable["utterances"][0]["written_text"] == project.utterances[0].text
        assert portable["utterances"][0]["spoken_text"] == "Texto hablado derivado"
        assert all(
            item["audio_relative_path"].startswith("audio/segments/")
            for item in portable["utterances"]
        )
        assert str(tmp_path) not in archive.read("speech-dialogue-project/manifest.json").decode()


def test_zip_refuses_overwrite_and_external_destination(make_wav, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project, master = _ready_project(make_wav, project_dir)
    destination = project_dir / "exports" / "project.zip"
    export_project_zip(project, project_dir, destination, master_wav=master)
    with pytest.raises(FileExistsError, match="confirma"):
        export_project_zip(project, project_dir, destination, master_wav=master)
    with pytest.raises(ValueError, match="dentro"):
        export_project_zip(project, project_dir, tmp_path / "outside.zip", master_wav=master)


def test_zip_without_audio_marks_pending_and_keeps_script(make_wav, tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    project, _ = _ready_project(make_wav, project_dir)
    project.utterances[0].status = "stale"
    project.utterances[1].status = "draft"
    project.utterances[1].audio_relative_path = None
    project.utterances[2].audio_relative_path = "audio/normalized/missing.wav"
    original = project.to_dict()
    destination = project_dir / "exports" / "pending.zip"

    _, manifest = export_project_zip(project, project_dir, destination)

    assert manifest["ready_utterance_ids"] == []
    assert manifest["pending_utterance_ids"] == [
        utterance.utterance_id for utterance in project.utterances
    ]
    assert project.to_dict() == original
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert "speech-dialogue-project/script/dialogue.txt" in names
        assert "speech-dialogue-project/audio/dialogue.wav" not in names
        assert not any("/audio/segments/" in name for name in names)


def test_individual_wav_reuses_only_the_selected_segment(make_wav, tmp_path: Path) -> None:
    selected = make_wav(tmp_path / "selected.wav", duration=0.1)
    other = make_wav(tmp_path / "other.wav", duration=0.35)

    exported = individual_wav_source(selected)

    assert exported == selected
    assert exported.read_bytes() == selected.read_bytes()
    assert exported.read_bytes() != other.read_bytes()
    assert exported.read_bytes()[:4] == b"RIFF"
    assert exported.read_bytes()[8:12] == b"WAVE"


@pytest.mark.skipif(not has_ffmpeg(), reason="FFmpeg no disponible")
def test_individual_mp3_contains_only_one_segment_and_is_cached(
    make_wav, tmp_path: Path
) -> None:
    project = DialogueProject.sample()
    selected = make_wav(tmp_path / "selected.wav", duration=0.1)
    other = make_wav(tmp_path / "other.wav", duration=0.4)
    utterance_id = project.utterances[0].utterance_id

    mp3, reused = prepare_individual_mp3(selected, tmp_path / "temporary", utterance_id)
    first_mtime = mp3.stat().st_mtime_ns
    cached, reused_cached = prepare_individual_mp3(
        selected, tmp_path / "temporary", utterance_id
    )

    assert not reused
    assert reused_cached
    assert cached == mp3
    assert cached.stat().st_mtime_ns == first_mtime
    assert is_mp3_file(cached)
    assert cached.read_bytes()[:3] == b"ID3" or cached.read_bytes()[:1] == b"\xff"
    assert probe_audio(cached).codec == "mp3"
    assert probe_audio(cached).duration_seconds == pytest.approx(0.1, abs=0.05)
    assert probe_audio(cached).duration_seconds < probe_audio(other).duration_seconds


def test_individual_export_names_paths_and_keys_are_safe_and_uuid_scoped(tmp_path: Path) -> None:
    project = DialogueProject.sample()
    first, second = project.utterances[:2]
    first_name = individual_export_filename(
        "../Álgebra / Redes", first.order, "Estudiante: uno", "wav"
    )
    second_name = individual_export_filename(
        "../Álgebra / Redes", second.order, "Estudiante: uno", ".mp3"
    )
    digest = "a" * 64
    first_path = individual_mp3_path(tmp_path, first.utterance_id, digest)
    second_path = individual_mp3_path(tmp_path, second.utterance_id, digest)

    assert first_name == "algebra_redes_intervencion_01_estudiante_uno.wav"
    assert second_name == "algebra_redes_intervencion_02_estudiante_uno.mp3"
    assert "/" not in first_name and "\\" not in first_name and ".." not in first_name
    assert first_path != second_path
    assert tmp_path.resolve() in first_path.parents
    assert individual_export_widget_key(first.utterance_id, "wav") != (
        individual_export_widget_key(second.utterance_id, "wav")
    )


def test_reordering_keeps_uuid_export_association_and_duplicate_has_new_identity(
    tmp_path: Path,
) -> None:
    project = DialogueProject.sample()
    source = project.utterances[0]
    digest = "b" * 64
    original_path = individual_mp3_path(tmp_path, source.utterance_id, digest)

    move_utterance(project, source.utterance_id, 1)
    moved = next(item for item in project.utterances if item.utterance_id == source.utterance_id)
    duplicate = duplicate_utterance(project, source.utterance_id)

    assert moved.order == 2
    assert individual_mp3_path(tmp_path, moved.utterance_id, digest) == original_path
    assert duplicate.utterance_id != source.utterance_id
    assert duplicate.audio_relative_path is None
    assert individual_export_widget_key(duplicate.utterance_id, "mp3") != (
        individual_export_widget_key(source.utterance_id, "mp3")
    )


def test_individual_mp3_failure_is_reported_without_partial_artifact(
    monkeypatch, make_wav, tmp_path: Path
) -> None:
    project = DialogueProject.new()
    source = make_wav(tmp_path / "source.wav")

    def unavailable(*_args, **_kwargs):
        raise AudioError("FFmpeg no está disponible; el WAV sigue operativo")

    monkeypatch.setattr("dialogue_studio.export.export_mp3", unavailable)

    with pytest.raises(AudioError, match="WAV sigue operativo"):
        prepare_individual_mp3(
            source,
            tmp_path / "temporary",
            project.utterances[0].utterance_id,
        )
    assert not list((tmp_path / "temporary").rglob("*.mp3"))
