from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dialogue_studio.ui as ui
from dialogue_studio.audio import AudioInfo
from dialogue_studio.paths import AppPaths
from dialogue_studio.speechnote import TTSModel


def _button(app: AppTest, label: str, occurrence: int = 0):
    return [button for button in app.button if button.label == label][occurrence]


def test_app_loads_sample_edits_adds_and_reorders_without_synthesis(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    paths = AppPaths(tmp_path / "Música")
    diagnostics = {
        "flatpak": True,
        "installed": True,
        "open": True,
        "ffmpeg": True,
        "models": [
            TTSModel("es_piper_mx_claude_high", "Profesor"),
            TTSModel("es_piper_es_sharvard_medium_1", "Estudiante"),
        ],
        "active": None,
        "error": None,
    }
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: diagnostics)

    def fake_generate(project, project_dir, utterance_id, controlled_root):
        calls.append(utterance_id)
        utterance = next(item for item in project.utterances if item.utterance_id == utterance_id)
        utterance.status = "ready"
        utterance.duration_seconds = 0.25
        return utterance

    monkeypatch.setattr(ui, "generate_utterance", fake_generate)
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert calls == []
    assert app.title[0].value == "Diálogo sin título"

    _button(app, "Ejemplo").click().run()
    assert not app.exception
    assert app.title[0].value == "Cómo aprende una red neuronal"
    assert len(app.session_state.project.speakers) == 2
    assert len(app.session_state.project.utterances) == 3
    assert calls == []

    first_id = app.session_state.project.utterances[0].utterance_id
    first_text = next(area for area in app.text_area if area.key == f"utterance-text-{first_id}")
    first_text.set_value("Texto editado").run()
    assert app.session_state.project.utterances[0].text == "Texto editado"
    assert calls == []

    _button(app, "↓", 0).click().run()
    assert app.session_state.project.utterances[1].utterance_id == first_id
    _button(app, "＋ Añadir intervención").click().run()
    assert len(app.session_state.project.utterances) == 4

    target = app.session_state.project.utterances[0]
    _button(app, "Generar", 0).click().run()
    assert calls == [target.utterance_id]
    assert app.session_state.project.utterances[0].status == "ready"


def test_session_state_contains_paths_not_audio_bytes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        ui.AppPaths,
        "discover",
        classmethod(lambda cls: AppPaths(tmp_path / "Music")),
    )
    monkeypatch.setattr(
        ui,
        "system_diagnostics",
        lambda: {
            "flatpak": False,
            "installed": False,
            "open": False,
            "ffmpeg": False,
            "models": [],
            "active": None,
            "error": None,
        },
    )
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert not any(isinstance(value, bytes) for value in app.session_state.filtered_state.values())
    generate_buttons = [button for button in app.button if button.label == "Generar"]
    assert generate_buttons and all(button.disabled for button in generate_buttons)


def test_generation_service_calls_synthesizer_once_and_keeps_old_preview_on_failure(
    make_wav, tmp_path: Path
) -> None:
    from dialogue_studio.models import DialogueProject
    from dialogue_studio.service import generate_utterance

    project = DialogueProject.sample()
    utterance = project.utterances[0]
    project_dir = tmp_path / "project"
    (project_dir / "audio" / "raw").mkdir(parents=True)
    (project_dir / "audio" / "normalized").mkdir(parents=True)
    calls = []

    def fake_synth(model_id, text, output, controlled_root, **kwargs):
        calls.append((model_id, text))
        make_wav(output)

    def fake_normalize(source, destination):
        make_wav(destination, duration=0.2)
        return AudioInfo("pcm_s16le", 48_000, 1, 0.2)

    generate_utterance(
        project,
        project_dir,
        utterance.utterance_id,
        tmp_path,
        synthesizer=fake_synth,
        normalizer=fake_normalize,
    )
    assert len(calls) == 1
    old_path = utterance.audio_relative_path

    def broken_synth(*args, **kwargs):
        raise RuntimeError("fallo controlado")

    with pytest.raises(RuntimeError, match="fallo controlado"):
        generate_utterance(
            project,
            project_dir,
            utterance.utterance_id,
            tmp_path,
            synthesizer=broken_synth,
            normalizer=fake_normalize,
        )
    assert utterance.status == "error"
    assert utterance.audio_relative_path == old_path
