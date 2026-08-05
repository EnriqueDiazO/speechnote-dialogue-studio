from __future__ import annotations

from pathlib import Path

from dialogue_studio.audio import AudioInfo, sha256_file
from dialogue_studio.models import DialogueProject
from dialogue_studio.recovery import (
    LEGACY_BUSY_MESSAGE,
    RECOVERABLE_MESSAGE,
    inspect_interrupted_synthesis,
    recover_interrupted_synthesis,
)


def _probe(path: Path) -> AudioInfo:
    if path.read_bytes()[:4] != b"RIFF":
        raise RuntimeError("invalid WAV")
    return AudioInfo("pcm_s16le", 48_000, 1, 0.1)


def _project_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "project"
    (directory / "audio" / "normalized").mkdir(parents=True)
    return directory


def test_generating_with_valid_matching_wav_becomes_ready(make_wav, tmp_path: Path) -> None:
    project = DialogueProject.new()
    utterance = project.utterances[0]
    directory = _project_directory(tmp_path)
    relative = f"audio/normalized/001-{utterance.utterance_id}.wav"
    wav = make_wav(directory / relative)
    utterance.status = "generating"
    utterance.audio_relative_path = relative
    original_text = utterance.text
    speaker_ids = [speaker.speaker_id for speaker in project.speakers]

    report = recover_interrupted_synthesis(project, directory, probe=_probe)

    assert report.changed
    assert report.recovered_ready == 1
    assert utterance.status == "ready"
    assert utterance.duration_seconds == 0.1
    assert utterance.sha256 == sha256_file(wav)
    assert utterance.error_message is None
    assert utterance.text == original_text
    assert [speaker.speaker_id for speaker in project.speakers] == speaker_ids


def test_generating_with_missing_wav_becomes_recoverable_stale(tmp_path: Path) -> None:
    project = DialogueProject.new()
    utterance = project.utterances[0]
    directory = _project_directory(tmp_path)
    utterance.status = "generating"
    utterance.audio_relative_path = f"audio/normalized/001-{utterance.utterance_id}.wav"

    report = recover_interrupted_synthesis(project, directory, probe=_probe)

    assert report.converted_recoverable == 1
    assert utterance.status == "stale"
    assert utterance.audio_relative_path is None
    assert utterance.error_message == RECOVERABLE_MESSAGE


def test_corrupt_wav_is_preserved_as_partial_and_not_playable(tmp_path: Path) -> None:
    project = DialogueProject.new()
    utterance = project.utterances[0]
    directory = _project_directory(tmp_path)
    relative = f"audio/normalized/001-{utterance.utterance_id}.wav"
    corrupt = directory / relative
    payload = b"corrupt-but-preserved" * 10
    corrupt.write_bytes(payload)
    utterance.status = "generating"
    utterance.audio_relative_path = relative

    report = recover_interrupted_synthesis(project, directory, probe=_probe)

    assert report.preserved_partial == 1
    assert utterance.status == "stale"
    assert utterance.audio_relative_path is None
    assert not corrupt.exists()
    partials = list((directory / "audio" / "recovery").glob("*.partial"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == payload


def test_legacy_busy_errors_recover_valid_audio_and_missing_output(
    make_wav, tmp_path: Path
) -> None:
    project = DialogueProject.sample()
    directory = _project_directory(tmp_path)
    first, second, third = project.utterances
    for utterance in (first, second):
        utterance.status = "error"
        utterance.error_message = f"{LEGACY_BUSY_MESSAGE}."
    first.audio_relative_path = f"audio/normalized/001-{first.utterance_id}.wav"
    make_wav(directory / first.audio_relative_path)
    second.audio_relative_path = None
    third.status = "error"
    third.error_message = "Otro error legítimo"
    original_order = [utterance.utterance_id for utterance in project.utterances]

    report = recover_interrupted_synthesis(project, directory, probe=_probe)

    assert report.affected_count == 2
    assert first.status == "ready"
    assert second.status == "stale"
    assert third.status == "error"
    assert [utterance.utterance_id for utterance in project.utterances] == original_order


def test_recovery_inspection_is_read_only_and_recovery_is_idempotent(
    make_wav, tmp_path: Path
) -> None:
    project = DialogueProject.new()
    directory = _project_directory(tmp_path)
    utterance = project.utterances[0]
    relative = f"audio/normalized/001-{utterance.utterance_id}.wav"
    make_wav(directory / relative)
    utterance.status = "generating"
    utterance.audio_relative_path = relative
    before = project.to_dict()

    inspection = inspect_interrupted_synthesis(project, directory, probe=_probe)
    assert inspection.affected_count == 1
    assert project.to_dict() == before
    recover_interrupted_synthesis(project, directory, probe=_probe)
    repeated = recover_interrupted_synthesis(project, directory, probe=_probe)
    assert not repeated.changed
    assert repeated.affected_count == 0
