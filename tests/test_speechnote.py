from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import dialogue_studio.speechnote as sn

MODEL_OUTPUT = """Available STT models: 13
    es_piper_es_sharvard_medium_1 "Español (Piper Sharvard Medium Female) / es"
    es_piper_mx_claude_high "Español mexicano (Piper Claude High) / es"
"""


def completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_model_parser_ignores_incorrect_header_and_finds_mexican_voice() -> None:
    models = sn.parse_tts_models(MODEL_OUTPUT)
    assert [model.model_id for model in models] == [
        "es_piper_es_sharvard_medium_1",
        "es_piper_mx_claude_high",
    ]
    assert "mexicano" in models[1].label


def test_list_and_active_models_use_argument_lists_without_shell() -> None:
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return completed(MODEL_OUTPUT)

    assert len(sn.list_tts_models(runner=runner)) == 2
    assert sn.get_active_tts_model(runner=runner).model_id == "es_piper_es_sharvard_medium_1"
    assert calls[0][0] == [
        "flatpak",
        "run",
        sn.APP_ID,
        "--print-available-models",
        "tts",
    ]
    assert all("shell" not in kwargs for _, kwargs in calls)


def test_synthesis_builds_exact_safe_command(monkeypatch, tmp_path: Path) -> None:
    calls = []
    waited = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return completed()

    monkeypatch.setattr(sn, "wait_for_wave", lambda path, **kwargs: waited.append(path))
    output = tmp_path / "root" / "audio" / "result.wav"
    sn.synthesize_text("voice_id", "Texto seguro", output, tmp_path / "root", runner=runner)
    assert calls[0][0] == [
        "flatpak",
        "run",
        sn.APP_ID,
        "--action",
        "start-reading-text",
        "--id",
        "voice_id",
        "--text",
        "Texto seguro",
        "--output-file",
        str(output),
    ]
    assert "shell" not in calls[0][1]
    assert waited == [output]
    assert not sn.synthesis_lock_active()


def test_external_invocation_disabled_has_instructions(tmp_path: Path) -> None:
    def runner(args, **kwargs):
        return completed(stderr="Action invocation is not enabled in settings", returncode=1)

    with pytest.raises(sn.SpeechNoteError, match="Ajustes"):
        sn.synthesize_text("voice", "Hola", tmp_path / "out.wav", tmp_path, runner=runner)
    assert not sn.synthesis_lock_active()


def test_wait_for_wave_valid_invalid_and_timeout(make_wav, tmp_path: Path) -> None:
    valid = make_wav(tmp_path / "valid.wav")
    probes = []
    sn.wait_for_wave(
        valid,
        timeout=0.1,
        stable_seconds=0,
        poll_interval=0.001,
        probe=lambda path: probes.append(path),
    )
    assert probes == [valid]
    invalid = tmp_path / "invalid.wav"
    invalid.write_bytes(b"not-wave" * 10)
    with pytest.raises(sn.SpeechNoteError, match="WAV inválido"):
        sn.wait_for_wave(invalid, timeout=0.1, stable_seconds=0, poll_interval=0.001)
    with pytest.raises(sn.SpeechNoteError, match="no creó"):
        sn.wait_for_wave(
            tmp_path / "missing.wav",
            timeout=0.005,
            stable_seconds=0,
            poll_interval=0.001,
        )


def test_synthesis_rejects_empty_text_bad_model_and_outside_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vacío"):
        sn.synthesize_text("voice", " ", tmp_path / "out.wav", tmp_path)
    with pytest.raises(ValueError, match="voz válida"):
        sn.synthesize_text("bad model", "Hola", tmp_path / "out.wav", tmp_path)
    with pytest.raises(ValueError, match="controlada"):
        sn.synthesize_text("voice", "Hola", tmp_path / "out.wav", tmp_path / "other")


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(["flatpak"], 1),
        RuntimeError("unexpected"),
    ],
    ids=["timeout", "unexpected"],
)
def test_command_failures_release_module_lock(failure: Exception, tmp_path: Path) -> None:
    def runner(args, **kwargs):
        raise failure

    with pytest.raises((sn.SpeechNoteError, RuntimeError)):
        sn.synthesize_text(
            "voice",
            "Hola",
            tmp_path / "output.wav",
            tmp_path,
            runner=runner,
        )
    assert not sn.synthesis_lock_active()


@pytest.mark.parametrize(
    "failure",
    [
        sn.SpeechNoteError("WAV inválido"),
        RuntimeError("ffprobe falló"),
    ],
    ids=["invalid-wav", "ffprobe"],
)
def test_output_validation_failures_release_module_lock(
    monkeypatch, failure: Exception, tmp_path: Path
) -> None:
    def fail_wait(*args, **kwargs):
        raise failure

    monkeypatch.setattr(sn, "wait_for_wave", fail_wait)

    with pytest.raises(type(failure)):
        sn.synthesize_text(
            "voice",
            "Hola",
            tmp_path / "output.wav",
            tmp_path,
            runner=lambda *args, **kwargs: completed(),
        )
    assert not sn.synthesis_lock_active()
