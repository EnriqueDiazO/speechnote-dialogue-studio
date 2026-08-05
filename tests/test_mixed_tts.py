from __future__ import annotations

from pathlib import Path

import pytest

from dialogue_studio.audio import probe_audio
from dialogue_studio.models import DialogueProject, SpeakerTTSConfig
from dialogue_studio.qwen_client import QwenClientError, synthesize_qwen_text
from dialogue_studio.service import (
    add_utterance,
    generate_utterance,
    update_speaker_tts,
)
from dialogue_studio.synthesis import SynthesisBusyError


def test_mixed_speechnote_and_qwen_segments_normalize_to_project_format(
    make_wav, tmp_path: Path
) -> None:
    project = DialogueProject.new()
    first = project.utterances[0]
    first.text = "Profesor con Speech Note"
    second = add_utterance(project, project.speakers[1].speaker_id, "Estudiante con Qwen")
    update_speaker_tts(
        project,
        project.speakers[1].speaker_id,
        SpeakerTTSConfig(
            provider="qwen",
            voice_id="vivian",
            voice_label="Vivian",
            language="spanish",
            generation_options={"seed": 9, "temperature": 0.8},
        ),
    )
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    speech_calls = []
    qwen_calls = []

    def speech(model_id, text, output, controlled_root, **_kwargs):
        speech_calls.append((model_id, text, controlled_root))
        make_wav(output, rate=48_000, duration=0.1)

    def qwen(voice, text, language, options, output):
        qwen_calls.append((voice, text, language, options))
        make_wav(output, rate=24_000, duration=0.15)

    generate_utterance(
        project,
        project_dir,
        first.utterance_id,
        tmp_path,
        synthesizer=speech,
        qwen_synthesizer=qwen,
    )
    generate_utterance(
        project,
        project_dir,
        second.utterance_id,
        tmp_path,
        synthesizer=speech,
        qwen_synthesizer=qwen,
    )
    assert len(speech_calls) == 1
    assert len(qwen_calls) == 1
    assert qwen_calls[0][0:3] == ("vivian", "Estudiante con Qwen", "spanish")
    assert qwen_calls[0][3]["max_new_tokens"] == 8192
    assert qwen_calls[0][3]["temperature"] == 0.8
    for utterance in (first, second):
        assert utterance.status == "ready"
        assert utterance.audio_fingerprint
        info = probe_audio(project_dir / utterance.audio_relative_path)
        assert info.is_master_format


class FakeQwenClient:
    def __init__(self, make_wav, *, rate: int = 24_000) -> None:
        self.make_wav = make_wav
        self.rate = rate
        self.payload = None

    def synthesize(self, **kwargs):
        self.payload = kwargs
        self.make_wav(kwargs["output_path"], rate=self.rate, duration=0.1)
        return {"output_path": str(kwargs["output_path"]), "elapsed_seconds": 0.1}


def test_qwen_adapter_validates_native_wav_and_sends_no_instruct(make_wav, tmp_path: Path) -> None:
    client = FakeQwenClient(make_wav)
    output = tmp_path / "native.wav"
    synthesize_qwen_text(
        "serena",
        "Hola",
        "spanish",
        {"seed": 1, "temperature": 0.9},
        output,
        client=client,
    )
    assert client.payload["speaker"] == "serena"
    assert "instruct" not in client.payload
    assert probe_audio(output).sample_rate == 24_000

    wrong_rate = FakeQwenClient(make_wav, rate=48_000)
    with pytest.raises(QwenClientError, match="24000"):
        synthesize_qwen_text(
            "serena",
            "Hola",
            "spanish",
            {"seed": 1},
            tmp_path / "wrong.wav",
            client=wrong_rate,
        )


def test_qwen_gpu_busy_is_transient_not_a_persisted_generation_error(tmp_path: Path) -> None:
    class BusyClient:
        def synthesize(self, **_kwargs):
            raise QwenClientError("GPU ocupada", code="gpu_busy", retryable=True)

    with pytest.raises(SynthesisBusyError, match="real activa"):
        synthesize_qwen_text(
            "serena",
            "Hola",
            "spanish",
            {"seed": 1},
            tmp_path / "busy.wav",
            client=BusyClient(),
        )


def test_qwen_failure_is_recoverable_and_preserves_previous_audio(tmp_path: Path) -> None:
    project = DialogueProject.new()
    utterance = project.utterances[0]
    utterance.text = "Texto"
    update_speaker_tts(
        project,
        utterance.speaker_id,
        SpeakerTTSConfig(
            provider="qwen",
            voice_id="ryan",
            voice_label="Ryan",
            language="spanish",
        ),
    )
    utterance.status = "ready"
    utterance.audio_relative_path = "audio/normalized/old.wav"
    utterance.audio_fingerprint = "old-fingerprint"
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    def fail(*_args, **_kwargs):
        raise QwenClientError("Fallo controlado", code="generation_failed")

    with pytest.raises(QwenClientError, match="controlado"):
        generate_utterance(
            project,
            project_dir,
            utterance.utterance_id,
            tmp_path,
            qwen_synthesizer=fail,
        )
    assert utterance.status == "error"
    assert utterance.audio_relative_path == "audio/normalized/old.wav"
    assert utterance.audio_fingerprint == "old-fingerprint"
