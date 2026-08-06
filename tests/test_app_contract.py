from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dialogue_studio.ui as ui
from dialogue_studio.audio import AudioInfo
from dialogue_studio.paths import AppPaths
from dialogue_studio.qwen_preview import QwenPreview
from dialogue_studio.recovery import LEGACY_BUSY_MESSAGE
from dialogue_studio.speechnote import TTSModel
from dialogue_studio.synthesis import SynthesisCoordinator


def _button(app: AppTest, label: str, occurrence: int = 0):
    return [button for button in app.button if button.label == label][occurrence]


def _keyed(elements, key: str):
    return next(element for element in elements if element.key == key)


def _diagnostics(
    *, tts_available: bool, qwen_available: bool = False
) -> dict[str, object]:
    return {
        "flatpak": tts_available,
        "installed": tts_available,
        "open": tts_available,
        "external_actions_enabled": tts_available,
        "ffmpeg": True,
        "models": [
            TTSModel("es_piper_mx_claude_high", "Profesor"),
            TTSModel("es_piper_es_sharvard_medium_1", "Estudiante"),
        ],
        "active": None,
        "error": None if tts_available else "Speech Note no está disponible",
        "qwen": {
            "ok": qwen_available,
            "state": "idle" if qwen_available else "offline",
            "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "gpu": "Fake RTX" if qwen_available else None,
            "model_loaded": qwen_available,
            "vram_free_bytes": 4 * 1024**3 if qwen_available else None,
            "vram_total_bytes": 8 * 1024**3 if qwen_available else None,
            "supports_instruct": False,
        },
        "qwen_capabilities": {
            "speakers": [
                "aiden",
                "dylan",
                "eric",
                "ono_anna",
                "ryan",
                "serena",
                "sohee",
                "uncle_fu",
                "vivian",
            ],
            "languages": ["auto", "english", "spanish"],
            "supports_instruct": False,
            "supports_voice_design": False,
            "supports_voice_cloning": False,
            "supports_sampling_controls": True,
            "supports_speaker_selection": True,
            "supports_language_selection": True,
        },
    }


def test_app_loads_sample_edits_adds_and_reorders_without_synthesis(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    paths = AppPaths(tmp_path / "Música")
    diagnostics = _diagnostics(tts_available=True)
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: diagnostics)

    def fake_generate(project, project_dir, utterance_id, controlled_root, **kwargs):
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
    assert not _keyed(app.button, f"generate-{first_id}").disabled
    assert not _button(app, "Generar pendientes").disabled

    _button(app, "↓", 0).click().run()
    assert app.session_state.project.utterances[1].utterance_id == first_id
    _button(app, "＋ Añadir intervención").click().run()
    assert len(app.session_state.project.utterances) == 4

    target = app.session_state.project.utterances[0]
    _button(app, "Generar", 0).click().run()
    assert calls == [target.utterance_id]
    assert app.session_state.project.utterances[0].status == "ready"


def test_editing_remains_available_when_speechnote_is_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Música")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=False))
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    utterance = app.session_state.project.utterances[0]

    assert not _button(app, "＋ Añadir intervención").disabled
    assert not _keyed(app.button, f"duplicate-{utterance.utterance_id}").disabled
    assert not _keyed(app.button, f"delete-{utterance.utterance_id}").disabled
    assert _keyed(app.button, f"up-{utterance.utterance_id}").disabled
    assert _keyed(app.button, f"down-{utterance.utterance_id}").disabled
    assert not _button(app, "Guardar proyecto").disabled
    assert not _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}").disabled
    assert not _keyed(app.selectbox, f"utterance-speaker-{utterance.utterance_id}").disabled
    assert _keyed(app.button, f"generate-{utterance.utterance_id}").disabled
    assert not _button(app, "Exportar proyecto ZIP").disabled

    utterance.audio_relative_path = "audio/normalized/previous.wav"
    utterance.status = "ready"
    app.run()
    assert _keyed(app.button, f"generate-{utterance.utterance_id}").label == "Regenerar"
    assert _keyed(app.button, f"generate-{utterance.utterance_id}").disabled

    _keyed(app.button, f"duplicate-{utterance.utterance_id}").click().run()
    assert len(app.session_state.project.utterances) == 2
    duplicate = app.session_state.project.utterances[1]
    _keyed(app.button, f"delete-{duplicate.utterance_id}").click().run()
    assert len(app.session_state.project.utterances) == 1
    _button(app, "＋ Añadir intervención").click().run()
    assert len(app.session_state.project.utterances) == 2


def test_speechnote_refresh_preserves_last_seen_unassigned_voices_when_closed(
    monkeypatch, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Música")
    available = _diagnostics(tts_available=True)
    available["models"] = [
        *available["models"],
        TTSModel("es_piper_mx_ald_extra", "Voz extra no asignada"),
    ]
    current = {"diagnostics": available}
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: current["diagnostics"])

    app = AppTest.from_file("app.py", default_timeout=30).run()
    speaker = app.session_state.project.speakers[0]
    configured_voice = speaker.tts_config.voice_id
    voice_menu = _keyed(app.selectbox, f"speaker-voice-{speaker.speaker_id}")
    assert "Voz extra no asignada" in voice_menu.options
    assert _button(app, "Actualizar voces de Speech Note")

    closed = _diagnostics(tts_available=False)
    closed["models"] = []
    current["diagnostics"] = closed
    app.run()

    speaker = app.session_state.project.speakers[0]
    voice_menu = _keyed(app.selectbox, f"speaker-voice-{speaker.speaker_id}")
    assert any(option.startswith("Voz extra no asignada") for option in voice_menu.options)
    assert speaker.tts_config.voice_id == configured_voice
    assert _button(app, "Actualizar voces de Speech Note")


def test_speaker_voice_uses_model_default_only_before_widget_state_exists(
    monkeypatch, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Música")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=True))
    calls = []
    update = ui.update_speaker_voice

    def tracked_update(project, speaker_id, model_id, model_label):
        calls.append((speaker_id, model_id))
        update(project, speaker_id, model_id, model_label)

    monkeypatch.setattr(ui, "update_speaker_voice", tracked_update)
    app = AppTest.from_file("app.py", default_timeout=30).run()
    speaker = app.session_state.project.speakers[0]
    key = f"speaker-voice-{speaker.speaker_id}"
    options = ["voice-a", "voice-b"]

    assert ui._selectbox_model_default({}, key, options, "voice-b") == {"index": 1}
    assert ui._selectbox_model_default({key: "voice-b"}, key, options, "voice-b") == {}

    voice = _keyed(app.selectbox, key)
    voice.select("es_piper_es_sharvard_medium_1").run()
    app.run()

    assert calls == [(speaker.speaker_id, "es_piper_es_sharvard_medium_1")]
    assert app.session_state.project.speakers[0].tts_config.voice_id == (
        "es_piper_es_sharvard_medium_1"
    )
    assert _keyed(app.selectbox, key).value == "es_piper_es_sharvard_medium_1"


def test_individual_export_controls_require_ready_audio_and_keep_uuid_keys(
    monkeypatch, make_wav, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Música")
    diagnostics = _diagnostics(tts_available=False)
    diagnostics["ffmpeg"] = False
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: diagnostics)
    app = AppTest.from_file("app.py", default_timeout=30).run()
    utterance = app.session_state.project.utterances[0]
    wav_key = f"intervention-export-wav-{utterance.utterance_id}"
    mp3_create_key = f"intervention-create-mp3-{utterance.utterance_id}"

    assert _keyed(app.download_button, wav_key).disabled
    assert _keyed(app.button, mp3_create_key).disabled
    assert any("genera primero" in item.value.lower() for item in app.caption)

    _button(app, "Guardar proyecto").click().run()
    project_dir = Path(app.session_state.project_dir)
    audio = make_wav(project_dir / "audio" / "normalized" / f"{utterance.utterance_id}.wav")
    utterance = app.session_state.project.utterances[0]
    utterance.audio_relative_path = audio.relative_to(project_dir).as_posix()
    utterance.sha256 = ui.sha256_file(audio)
    utterance.duration_seconds = 0.1
    utterance.status = "ready"
    app.run()

    assert not _keyed(app.download_button, wav_key).disabled
    assert _keyed(app.button, mp3_create_key).disabled
    assert any("necesita ffmpeg" in item.value.lower() for item in app.caption)

    _button(app, "＋ Añadir intervención").click().run()
    duplicate = app.session_state.project.utterances[1]
    assert duplicate.utterance_id != utterance.utterance_id
    assert _keyed(
        app.download_button, f"intervention-export-wav-{duplicate.utterance_id}"
    ).disabled
    _keyed(app.button, f"up-{duplicate.utterance_id}").click().run()
    assert not _keyed(app.download_button, wav_key).disabled


@pytest.mark.skipif(not ui.has_ffmpeg(), reason="FFmpeg no disponible")
def test_individual_mp3_button_converts_once_then_exposes_download(
    monkeypatch, make_wav, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Música")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=False))
    calls = []
    prepare = ui.prepare_individual_mp3

    def tracked_prepare(source, temporary_root, utterance_id):
        calls.append((source, utterance_id))
        return prepare(source, temporary_root, utterance_id)

    monkeypatch.setattr(ui, "prepare_individual_mp3", tracked_prepare)
    app = AppTest.from_file("app.py", default_timeout=30).run()
    utterance = app.session_state.project.utterances[0]
    _button(app, "Guardar proyecto").click().run()
    project_dir = Path(app.session_state.project_dir)
    audio = make_wav(project_dir / "audio" / "normalized" / f"{utterance.utterance_id}.wav")
    utterance = app.session_state.project.utterances[0]
    utterance.audio_relative_path = audio.relative_to(project_dir).as_posix()
    utterance.sha256 = ui.sha256_file(audio)
    utterance.duration_seconds = 0.1
    utterance.status = "ready"
    app.run()

    _keyed(app.button, f"intervention-create-mp3-{utterance.utterance_id}").click().run()

    mp3_key = f"intervention-export-mp3-{utterance.utterance_id}"
    assert calls == [(audio, utterance.utterance_id)]
    assert _keyed(app.download_button, mp3_key).label == "Descargar MP3"
    assert app.session_state.project.utterances[0].audio_relative_path == (
        audio.relative_to(project_dir).as_posix()
    )
    app.run()
    assert calls == [(audio, utterance.utterance_id)]
    assert _keyed(app.download_button, mp3_key)


def test_stale_legacy_busy_flag_recovers_without_locking_editing(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ui.AppPaths,
        "discover",
        classmethod(lambda cls: AppPaths(tmp_path / "Music")),
    )
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=False))
    app = AppTest.from_file("app.py", default_timeout=30).run()
    utterance = app.session_state.project.utterances[0]
    app.session_state.busy = True
    app.run()
    assert not app.exception
    assert "busy" not in app.session_state.filtered_state
    assert not _button(app, "＋ Añadir intervención").disabled
    assert not _keyed(app.button, f"duplicate-{utterance.utterance_id}").disabled
    assert not _keyed(app.button, f"delete-{utterance.utterance_id}").disabled


def test_ready_audio_plays_then_text_edit_marks_it_stale(
    monkeypatch, make_wav, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Música")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=True))
    app = AppTest.from_file("app.py", default_timeout=30).run()
    utterance = app.session_state.project.utterances[0]
    text_area = _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}")
    text_area.set_value("Audio vigente").run()
    _button(app, "Guardar proyecto").click().run()
    directory = Path(app.session_state.project_dir)
    relative = f"audio/normalized/{utterance.utterance_id}.wav"
    make_wav(directory / relative)
    utterance = app.session_state.project.utterances[0]
    utterance.audio_relative_path = relative
    utterance.duration_seconds = 0.1
    utterance.status = "ready"
    app.run()

    assert any(expander.label == "Escuchar" for expander in app.expander)
    assert not _button(app, "Construir diálogo").disabled
    assert not _keyed(app.button, f"generate-{utterance.utterance_id}").disabled
    _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}").set_value(
        "Audio ahora desactualizado"
    ).run()
    assert app.session_state.project.utterances[0].status == "stale"
    assert not _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}").disabled
    assert not any(expander.label == "Escuchar" for expander in app.expander)
    assert _button(app, "Construir diálogo").disabled
    regenerate = _keyed(app.button, f"generate-{utterance.utterance_id}")
    assert regenerate.label == "Regenerar"
    assert not regenerate.disabled


def test_error_and_stale_are_regenerable_without_showing_a_live_lock(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ui.AppPaths,
        "discover",
        classmethod(lambda cls: AppPaths(tmp_path / "Music")),
    )
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=True))
    app = AppTest.from_file("app.py", default_timeout=30).run()
    utterance = app.session_state.project.utterances[0]
    _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}").set_value("Texto").run()
    utterance = app.session_state.project.utterances[0]
    utterance.status = "error"
    utterance.error_message = "Fallo anterior"
    app.run()
    regenerate = _keyed(app.button, f"generate-{utterance.utterance_id}")
    assert regenerate.label == "Regenerar"
    assert not regenerate.disabled
    assert not any("síntesis en curso" in item.value.lower() for item in app.error)

    utterance.status = "stale"
    app.run()
    regenerate = _keyed(app.button, f"generate-{utterance.utterance_id}")
    assert not regenerate.disabled


def test_manual_recovery_removes_persisted_busy_message(
    monkeypatch, make_wav, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Music")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=False))
    app = AppTest.from_file("app.py", default_timeout=30).run()
    utterance = app.session_state.project.utterances[0]
    _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}").set_value("Texto").run()
    _button(app, "Guardar proyecto").click().run()
    directory = Path(app.session_state.project_dir)
    relative = f"audio/normalized/001-{utterance.utterance_id}.wav"
    make_wav(directory / relative)
    utterance = app.session_state.project.utterances[0]
    utterance.status = "error"
    utterance.error_message = LEGACY_BUSY_MESSAGE
    utterance.audio_relative_path = relative
    app.run()

    assert _button(app, "Recuperar síntesis interrumpida")
    assert not any("ya hay una síntesis" in item.value.lower() for item in app.error)
    _button(app, "Recuperar síntesis interrumpida").click().run()
    assert app.session_state.project.utterances[0].status == "ready"
    assert not [
        button for button in app.button if button.label == "Recuperar síntesis interrumpida"
    ]


def test_only_real_active_synthesis_shows_busy_state(monkeypatch, tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "Music")
    coordinator = SynthesisCoordinator()
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=True))
    monkeypatch.setattr(ui, "GLOBAL_SYNTHESIS_COORDINATOR", coordinator)
    app = AppTest.from_file("app.py", default_timeout=30).run()
    utterance = app.session_state.project.utterances[0]
    active = coordinator.start(
        utterance.utterance_id,
        tmp_path / "active.wav",
        "another-session",
    )
    try:
        app.run()
        assert any("Sintetizando intervención 01" in item.value for item in app.info)
        assert _keyed(app.button, f"generate-{utterance.utterance_id}").disabled
        assert _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}").disabled
        assert not _button(app, "＋ Añadir intervención").disabled
    finally:
        coordinator.clear(active.job_token)


def test_move_boundaries_and_order_survive_rerun(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        ui.AppPaths,
        "discover",
        classmethod(lambda cls: AppPaths(tmp_path / "Music")),
    )
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=False))
    app = AppTest.from_file("app.py", default_timeout=30).run()
    _button(app, "Ejemplo").click().run()
    utterances = list(app.session_state.project.utterances)
    first, middle, last = utterances
    assert _keyed(app.button, f"up-{first.utterance_id}").disabled
    assert not _keyed(app.button, f"down-{first.utterance_id}").disabled
    assert not _keyed(app.button, f"up-{middle.utterance_id}").disabled
    assert not _keyed(app.button, f"down-{middle.utterance_id}").disabled
    assert not _keyed(app.button, f"up-{last.utterance_id}").disabled
    assert _keyed(app.button, f"down-{last.utterance_id}").disabled

    _keyed(app.button, f"down-{first.utterance_id}").click().run()
    assert [item.utterance_id for item in app.session_state.project.utterances] == [
        middle.utterance_id,
        first.utterance_id,
        last.utterance_id,
    ]
    app.run()
    assert [item.order for item in app.session_state.project.utterances] == [1, 2, 3]


def test_save_and_reopen_preserves_added_utterances(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        ui.AppPaths,
        "discover",
        classmethod(lambda cls: AppPaths(tmp_path / "Music")),
    )
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=False))
    app = AppTest.from_file("app.py", default_timeout=30).run()
    first = app.session_state.project.utterances[0]
    _keyed(app.text_area, f"utterance-text-{first.utterance_id}").set_value("Primera").run()
    _button(app, "＋ Añadir intervención").click().run()
    second = app.session_state.project.utterances[1]
    _keyed(app.text_area, f"utterance-text-{second.utterance_id}").set_value("Segunda").run()
    project_id = app.session_state.project.project_id
    _button(app, "Guardar proyecto").click().run()

    _button(app, "Nuevo").click().run()
    assert app.session_state.project.project_id != project_id
    _button(app, "Abrir seleccionado").click().run()
    assert app.session_state.project.project_id == project_id
    assert [item.text for item in app.session_state.project.utterances] == [
        "Primera",
        "Segunda",
    ]


def test_session_state_contains_paths_not_audio_bytes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        ui.AppPaths,
        "discover",
        classmethod(lambda cls: AppPaths(tmp_path / "Music")),
    )
    monkeypatch.setattr(
        ui,
        "system_diagnostics",
        lambda: _diagnostics(tts_available=False),
    )
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    assert not any(isinstance(value, bytes) for value in app.session_state.filtered_state.values())
    generate_buttons = [button for button in app.button if button.label == "Generar"]
    assert generate_buttons and all(button.disabled for button in generate_buttons)


def test_rendering_does_not_generate_or_modify_files(monkeypatch, tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "Music")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: _diagnostics(tts_available=True))

    def forbidden(*args, **kwargs):
        raise AssertionError("Renderizar no debe sintetizar")

    monkeypatch.setattr(ui, "generate_utterance", forbidden)
    monkeypatch.setattr(ui, "synthesize_text", forbidden)
    app = AppTest.from_file("app.py", default_timeout=30).run()
    before = sorted(
        path.relative_to(paths.root) for path in paths.root.rglob("*") if path.is_file()
    )
    app.run()
    after = sorted(path.relative_to(paths.root) for path in paths.root.rglob("*") if path.is_file())
    assert not app.exception
    assert before == after == []


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


def test_busy_error_does_not_overwrite_a_ready_utterance(make_wav, tmp_path: Path) -> None:
    from dialogue_studio.models import DialogueProject
    from dialogue_studio.service import generate_utterance
    from dialogue_studio.synthesis import SynthesisBusyError

    project = DialogueProject.new()
    utterance = project.utterances[0]
    utterance.text = "Texto"
    utterance.status = "ready"
    utterance.audio_relative_path = f"audio/normalized/{utterance.utterance_id}.wav"
    utterance.duration_seconds = 0.1
    utterance.sha256 = "old-hash"
    project_dir = tmp_path / "project"
    (project_dir / "audio" / "raw").mkdir(parents=True)
    (project_dir / "audio" / "normalized").mkdir(parents=True)
    make_wav(project_dir / utterance.audio_relative_path)

    def busy(*args, **kwargs):
        raise SynthesisBusyError("Hay una síntesis real activa")

    with pytest.raises(SynthesisBusyError):
        generate_utterance(
            project,
            project_dir,
            utterance.utterance_id,
            tmp_path,
            synthesizer=busy,
        )
    assert utterance.status == "ready"
    assert utterance.audio_relative_path.endswith(".wav")
    assert utterance.duration_seconds == 0.1
    assert utterance.sha256 == "old-hash"


def test_qwen_character_menu_exposes_real_capabilities_without_expression_controls(
    monkeypatch, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Music")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(
        ui,
        "system_diagnostics",
        lambda: _diagnostics(tts_available=False, qwen_available=True),
    )
    app = AppTest.from_file("app.py", default_timeout=30).run()
    speaker = app.session_state.project.speakers[0]
    _keyed(app.selectbox, f"speaker-provider-{speaker.speaker_id}").select("qwen").run()

    speaker = app.session_state.project.speakers[0]
    assert speaker.tts_config.provider == "qwen"
    voice = _keyed(app.selectbox, f"speaker-voice-{speaker.speaker_id}")
    assert voice.options == [
        "Aiden",
        "Dylan",
        "Eric",
        "Ono Anna",
        "Ryan",
        "Serena",
        "Sohee",
        "Uncle Fu",
        "Vivian",
    ]
    assert _keyed(app.selectbox, f"speaker-language-{speaker.speaker_id}").value == "spanish"
    assert any(item.label == "Temperatura" for item in app.number_input)
    active_labels = {
        item.label
        for collection in (app.selectbox, app.number_input, app.slider, app.text_area)
        for item in collection
    }
    assert not {"Emoción", "Estilo", "Ritmo", "Intensidad", "Pausas", "Claridad"} & active_labels
    assert any("no admite instrucciones" in item.value.lower() for item in app.info)


def test_qwen_sampling_and_utterance_override_persist_and_mark_stale(
    monkeypatch, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Music")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(
        ui,
        "system_diagnostics",
        lambda: _diagnostics(tts_available=True, qwen_available=True),
    )
    app = AppTest.from_file("app.py", default_timeout=30).run()
    speaker = app.session_state.project.speakers[0]
    utterance = app.session_state.project.utterances[0]
    _keyed(app.selectbox, f"speaker-provider-{speaker.speaker_id}").select("qwen").run()
    speaker = app.session_state.project.speakers[0]
    utterance = app.session_state.project.utterances[0]
    utterance.status = "ready"
    utterance.audio_relative_path = f"audio/normalized/{utterance.utterance_id}.wav"
    _keyed(app.selectbox, f"speaker-voice-{speaker.speaker_id}").select("serena").run()
    _keyed(app.number_input, f"speaker-qwen-{speaker.speaker_id}-temperature").set_value(
        0.7
    ).run()
    assert speaker.tts_config.voice_id == "serena"
    assert speaker.tts_config.generation_options["temperature"] == 0.7
    assert utterance.status == "stale"

    override_toggle = _keyed(
        app.checkbox, f"utterance-override-enabled-{utterance.utterance_id}"
    )
    override_toggle.check().run()
    _keyed(app.selectbox, f"utterance-qwen-voice-{utterance.utterance_id}").select(
        "vivian"
    ).run()
    assert utterance.tts_override is not None
    assert utterance.tts_override.provider is None
    assert utterance.tts_override.voice_id == "vivian"


def test_qwen_gallery_preview_assigns_voice_without_changing_utterance_audio(
    monkeypatch, make_wav, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Music")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(
        ui,
        "system_diagnostics",
        lambda: _diagnostics(tts_available=True, qwen_available=True),
    )
    app = AppTest.from_file("app.py", default_timeout=30).run()
    utterance = app.session_state.project.utterances[0]
    preview = make_wav(paths.temporary / "qwen-previews" / "preview.wav", rate=24_000)
    app.session_state.qwen_gallery_records = [
        QwenPreview(
            fingerprint="abc123",
            voice_id="vivian",
            language="spanish",
            path=preview,
            duration_seconds=0.1,
            elapsed_seconds=1.2,
            cached=False,
        )
    ]
    app.run()
    _keyed(app.button, "qwen-preview-assign-abc123").click().run()
    speaker = app.session_state.project.speakers[0]
    assert speaker.tts_config.provider == "qwen"
    assert speaker.tts_config.voice_id == "vivian"
    assert utterance.audio_relative_path is None


def test_qwen_only_project_can_generate_without_speechnote(
    monkeypatch, tmp_path: Path
) -> None:
    from dialogue_studio.models import SpeakerTTSConfig, effective_tts_config
    from dialogue_studio.service import update_speaker_tts

    paths = AppPaths(tmp_path / "Music")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(
        ui,
        "system_diagnostics",
        lambda: _diagnostics(tts_available=False, qwen_available=True),
    )
    calls = []

    def fake_generate(project, project_dir, utterance_id, controlled_root, **_kwargs):
        utterance = next(item for item in project.utterances if item.utterance_id == utterance_id)
        calls.append(effective_tts_config(project, utterance).provider)
        utterance.status = "ready"
        return utterance

    monkeypatch.setattr(ui, "generate_utterance", fake_generate)
    app = AppTest.from_file("app.py", default_timeout=30).run()
    project = app.session_state.project
    utterance = project.utterances[0]
    update_speaker_tts(
        project,
        utterance.speaker_id,
        SpeakerTTSConfig(
            provider="qwen",
            voice_id="serena",
            voice_label="Serena",
            language="spanish",
        ),
    )
    app.session_state.speaker_widget_resets = [utterance.speaker_id]
    app.run()
    _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}").set_value(
        "Generación local con Qwen"
    ).run()
    generate = _keyed(app.button, f"generate-{utterance.utterance_id}")
    assert not generate.disabled
    generate.click().run()
    assert calls == ["qwen"]


def test_blocked_qwen_keeps_speechnote_editing_and_generation_available(
    monkeypatch, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Music")
    diagnostics = _diagnostics(tts_available=True, qwen_available=True)
    diagnostics["qwen_preflight"] = {
        "allowed": False,
        "blockers": ["Se detectaron eventos Xid recientes"],
        "warnings": [],
        "recommended_actions": [],
    }
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: diagnostics)
    app = AppTest.from_file("app.py", default_timeout=30).run()
    utterance = app.session_state.project.utterances[0]
    text = _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}")
    text.set_value("Speech Note sigue disponible").run()
    assert not _keyed(app.button, f"generate-{utterance.utterance_id}").disabled
    speaker = app.session_state.project.speakers[0]
    _keyed(app.selectbox, f"speaker-provider-{speaker.speaker_id}").select("qwen").run()
    assert _keyed(app.button, f"generate-{utterance.utterance_id}").disabled
    assert any(
        "Generación bloqueada para proteger la sesión gráfica" in item.value
        for item in app.error
    )


def test_safe_qwen_requires_one_session_confirmation(monkeypatch, tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "Music")
    diagnostics = _diagnostics(tts_available=False, qwen_available=True)
    diagnostics["qwen_preflight"] = {
        "allowed": True,
        "gpu_name": "Fake RTX",
        "warnings": ["La GPU también maneja pantallas"],
        "blockers": [],
        "recommended_actions": [],
    }
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", lambda: diagnostics)
    app = AppTest.from_file("app.py", default_timeout=30).run()
    project = app.session_state.project
    utterance = project.utterances[0]
    speaker = project.speakers[0]
    _keyed(app.selectbox, f"speaker-provider-{speaker.speaker_id}").select("qwen").run()
    _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}").set_value("Hola").run()
    assert _keyed(app.button, f"generate-{utterance.utterance_id}").disabled
    _keyed(app.checkbox, "qwen-risk-confirmation").check().run()
    assert app.session_state.qwen_session_confirmed is True
    assert not _keyed(app.button, f"generate-{utterance.utterance_id}").disabled


def test_rendering_inherited_qwen_override_does_not_mark_ready_audio_stale(
    monkeypatch, tmp_path: Path
) -> None:
    from dialogue_studio.models import SpeakerTTSConfig, UtteranceTTSOverride
    from dialogue_studio.service import update_speaker_tts

    paths = AppPaths(tmp_path / "Music")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(
        ui,
        "system_diagnostics",
        lambda: _diagnostics(tts_available=True, qwen_available=True),
    )
    app = AppTest.from_file("app.py", default_timeout=30).run()
    project = app.session_state.project
    utterance = project.utterances[0]
    update_speaker_tts(
        project,
        utterance.speaker_id,
        SpeakerTTSConfig(
            provider="qwen",
            voice_id="serena",
            voice_label="Serena",
            language="spanish",
            generation_options={"seed": 1, "temperature": 0.9},
        ),
    )
    utterance.tts_override = UtteranceTTSOverride(voice_id="vivian")
    utterance.status = "ready"
    app.session_state.reset_project_widgets = True
    app.run()
    assert app.session_state.project.utterances[0].status == "ready"
    assert app.session_state.project.utterances[0].tts_override == UtteranceTTSOverride(
        voice_id="vivian"
    )
