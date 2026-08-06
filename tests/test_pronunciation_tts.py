from __future__ import annotations

from pathlib import Path

import pytest

from dialogue_studio.audio import AudioInfo
from dialogue_studio.models import DialogueProject, SpeakerTTSConfig
from dialogue_studio.pronunciation import PronunciationRule
from dialogue_studio.service import (
    PronunciationTransformationError,
    add_utterance,
    effective_pronunciation_result,
    generate_utterance,
    update_speaker_tts,
)


def test_speechnote_and_qwen_receive_the_exact_same_spoken_text(
    make_wav,
    tmp_path: Path,
) -> None:
    project = DialogueProject.new()
    written = "Qwen estudia θ_{t+1}=θ_t-η∇L(θ_t)."
    first = project.utterances[0]
    first.text = written
    second = add_utterance(project, project.speakers[1].speaker_id, written)
    update_speaker_tts(
        project,
        project.speakers[1].speaker_id,
        SpeakerTTSConfig(
            provider="qwen",
            voice_id="vivian",
            voice_label="Vivian",
            language="spanish",
        ),
    )
    global_rule = PronunciationRule.create(
        scope="global",
        language="es",
        kind="literal",
        pattern="Qwen",
        replacement="cuen",
    )
    global_rules = [global_rule]
    expected = effective_pronunciation_result(
        project,
        first,
        global_rules=global_rules,
    ).spoken_text
    speech_texts: list[str] = []
    qwen_texts: list[str] = []

    def speech(_voice, text, output, _root, **_kwargs):
        speech_texts.append(text)
        make_wav(output)

    def qwen(_voice, text, _language, _options, output):
        qwen_texts.append(text)
        make_wav(output, rate=24_000)

    def normalize(_source, destination):
        make_wav(destination)
        return AudioInfo("pcm_s16le", 48_000, 1, 0.1)

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    for utterance in (first, second):
        generate_utterance(
            project,
            project_dir,
            utterance.utterance_id,
            tmp_path,
            synthesizer=speech,
            qwen_synthesizer=qwen,
            normalizer=normalize,
            global_rules=global_rules,
        )

    assert speech_texts == qwen_texts == [expected]
    assert expected != written
    assert first.text == second.text == written
    assert first.spoken_text == second.spoken_text == expected
    assert first.written_text_hash and first.spoken_text_hash
    assert first.pronunciation_rules_hash
    assert first.pronunciation_engine_version == "1.0"
    assert global_rules[0].usage_count == 2


def test_manual_override_is_the_only_text_sent_to_provider(make_wav, tmp_path: Path) -> None:
    project = DialogueProject.new()
    utterance = project.utterances[0]
    utterance.text = "texto escrito"
    utterance.manual_spoken_text_override = "lectura manual exacta"
    received: list[str] = []

    def speech(_voice, text, output, _root, **_kwargs):
        received.append(text)
        make_wav(output)

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    generate_utterance(
        project,
        project_dir,
        utterance.utterance_id,
        tmp_path,
        synthesizer=speech,
    )
    assert received == ["lectura manual exacta"]
    assert utterance.text == "texto escrito"
    assert utterance.applied_pronunciation_rule_ids == ["manual-utterance-override"]


class BrokenPronunciationEngine:
    def transform(self, *_args, **_kwargs):
        raise ValueError("fallo de transformación controlado")


def test_transformation_failure_never_locks_and_fallback_requires_confirmation(
    make_wav,
    tmp_path: Path,
) -> None:
    project = DialogueProject.new()
    utterance = project.utterances[0]
    utterance.text = "Texto original"
    utterance.status = "ready"
    utterance.audio_relative_path = "audio/normalized/old.wav"
    received: list[str] = []

    def speech(_voice, text, output, _root, **_kwargs):
        received.append(text)
        make_wav(output)

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    with pytest.raises(PronunciationTransformationError, match="fallback"):
        generate_utterance(
            project,
            project_dir,
            utterance.utterance_id,
            tmp_path,
            synthesizer=speech,
            pronunciation_engine=BrokenPronunciationEngine(),  # type: ignore[arg-type]
        )
    assert utterance.status == "ready"
    assert utterance.audio_relative_path == "audio/normalized/old.wav"
    assert received == []

    generate_utterance(
        project,
        project_dir,
        utterance.utterance_id,
        tmp_path,
        synthesizer=speech,
        pronunciation_engine=BrokenPronunciationEngine(),  # type: ignore[arg-type]
        allow_pronunciation_fallback=True,
    )
    assert received == ["Texto original"]
    assert utterance.status == "ready"
    assert any(
        warning["code"] == "explicit_written_text_fallback"
        for warning in utterance.pronunciation_warnings
    )
