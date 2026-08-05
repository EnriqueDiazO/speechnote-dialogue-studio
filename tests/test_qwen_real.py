from __future__ import annotations

import json
import os
import wave
from pathlib import Path
from uuid import uuid4

import pytest

from dialogue_studio.audio import export_mp3, normalize_audio, probe_audio, sha256_file
from dialogue_studio.models import (
    DialogueProject,
    SpeakerTTSConfig,
    UtteranceTTSOverride,
)
from dialogue_studio.paths import AppPaths, safe_write_path
from dialogue_studio.qwen_client import QwenBackendManager, synthesize_qwen_text
from dialogue_studio.service import add_speaker, add_utterance, build_master
from dialogue_studio.storage import atomic_write_text, deterministic_json

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN_REAL") != "1",
    reason="Requiere el backend Qwen real y una GPU CUDA",
)


def _synthetic_master_segment(path: Path, duration: float = 0.1) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(b"\x00\x00" * round(48_000 * duration))


def test_real_qwen_three_voices_mixed_master_and_restart() -> None:
    paths = AppPaths.discover()
    paths.ensure()
    manager = QwenBackendManager(paths)
    manager.start()
    token = uuid4().hex[:12]
    validation_root = safe_write_path(paths.root, f"temporary/qwen-real-{token}")
    validation_root.mkdir(mode=0o700, parents=True)
    raw_root = validation_root / "native"
    normalized_root = validation_root / "project" / "audio" / "normalized"
    raw_root.mkdir(mode=0o700)
    normalized_root.mkdir(mode=0o700, parents=True)
    text = "Hola. Esta voz participa en una prueba multivoz en español."
    options: dict[str, int | float] = {
        "seed": 2026,
        "max_new_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }
    before = manager.client.health()
    native_paths: dict[str, Path] = {}
    normalized_paths: dict[str, Path] = {}
    for voice in ("serena", "vivian", "ryan"):
        native = raw_root / f"{voice}.wav"
        synthesize_qwen_text(
            voice,
            text,
            "spanish",
            options,
            native,
            client=manager.client,
        )
        native_info = probe_audio(native)
        assert native_info.codec == "pcm_s16le"
        assert native_info.sample_rate == 24_000
        assert native_info.channels == 1
        assert native_info.duration_seconds > 0
        normalized = normalized_root / f"{voice}.wav"
        normalized_info = normalize_audio(native, normalized)
        assert normalized_info.is_master_format
        native_paths[voice] = native
        normalized_paths[voice] = normalized
    after = manager.client.health()
    expected_loads = int(before.get("load_count", 0)) + (0 if before.get("model_loaded") else 1)
    assert after["load_count"] == expected_loads
    assert after["model_loaded"] is True
    assert after["state"] == "idle"
    assert after["last_error"] is None
    assert len({sha256_file(path) for path in native_paths.values()}) == 3

    project_dir = validation_root / "project"
    project = DialogueProject.new("Validación real mixta")
    speech = project.utterances[0]
    speech.text = "Segmento de referencia de Speech Note."
    speech_path = normalized_root / "speechnote-reference.wav"
    _synthetic_master_segment(speech_path)
    speech.audio_relative_path = speech_path.relative_to(project_dir).as_posix()
    speech.duration_seconds = probe_audio(speech_path).duration_seconds
    speech.sha256 = sha256_file(speech_path)
    speech.status = "ready"

    qwen_speaker = add_speaker(
        project,
        "Qwen",
        "serena",
        "Serena",
        tts=SpeakerTTSConfig(
            provider="qwen",
            voice_id="serena",
            voice_label="Serena",
            language="spanish",
            generation_options=options,
        ),
    )
    for voice in ("serena", "vivian", "ryan"):
        utterance = add_utterance(project, qwen_speaker.speaker_id, text)
        utterance.tts_override = UtteranceTTSOverride(voice_id=voice)
        utterance.audio_relative_path = normalized_paths[voice].relative_to(project_dir).as_posix()
        utterance.duration_seconds = probe_audio(normalized_paths[voice]).duration_seconds
        utterance.sha256 = sha256_file(normalized_paths[voice])
        utterance.status = "ready"
    master, master_info = build_master(project, project_dir)
    assert master_info.is_master_format
    mp3 = project_dir / "exports" / "dialogue.mp3"
    mp3_info = export_mp3(master, mp3)
    assert mp3_info.sample_rate == 48_000
    assert mp3_info.channels == 1
    project_file = project_dir / "project.json"
    atomic_write_text(project_file, deterministic_json(project.to_dict()))
    reloaded = DialogueProject.from_dict(json.loads(project_file.read_text(encoding="utf-8")))
    assert len(reloaded.utterances) == 4
    assert [item.status for item in reloaded.utterances] == ["ready"] * 4

    stopped = manager.stop()
    assert stopped["state"] == "offline"
    assert not manager.pid_file.exists()
    restarted = manager.start()
    assert restarted["state"] == "idle"
    assert restarted["model_loaded"] is False
