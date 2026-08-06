from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import dialogue_studio.ui as ui
from dialogue_studio.paths import AppPaths


def _keyed(elements, key: str):
    return next(element for element in elements if element.key == key)


def _diagnostics() -> dict[str, object]:
    return {
        "flatpak": False,
        "installed": False,
        "open": False,
        "external_actions_enabled": False,
        "ffmpeg": False,
        "models": [],
        "active": None,
        "error": "TTS no disponible durante la prueba",
        "qwen": {"ok": False, "state": "offline"},
        "qwen_capabilities": {
            "speakers": ["serena"],
            "languages": ["auto", "spanish", "english"],
        },
    }


def _app(monkeypatch, tmp_path: Path) -> AppTest:
    paths = AppPaths(tmp_path / "Music")
    monkeypatch.setattr(ui.AppPaths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(ui, "system_diagnostics", _diagnostics)
    return AppTest.from_file("app.py", default_timeout=30).run()


def test_preview_trace_warning_and_quick_project_rule(monkeypatch, tmp_path: Path) -> None:
    app = _app(monkeypatch, tmp_path)
    assert not app.exception
    _keyed(app.text_area, "pronunciation-preview-written").set_value(
        r"La ecuación $\frac{\partial L}{\partial x}$ usa MOFA2"
    ).run()
    _keyed(app.button, "pronunciation-preview-transform").click().run()

    result = app.session_state.pronunciation_preview_result
    assert "derivada parcial" in result.spoken_text
    assert result.unsupported_fragments == ("MOFA2",)
    assert any("por revisar" in warning.value for warning in app.warning)

    _keyed(app.text_input, "pronunciation-preview-quick-term").set_value("MOFA2").run()
    _keyed(app.text_input, "pronunciation-preview-quick-spoken").set_value(
        "mofa dos"
    ).run()
    _keyed(app.button, "pronunciation-preview-quick-add-project").click().run()
    saved_rules = app.session_state.project.pronunciation_rules
    assert [(rule.pattern, rule.replacement) for rule in saved_rules] == [
        ("MOFA2", "mofa dos")
    ]
    assert not app.exception


def test_global_and_project_rule_editors_persist_without_session_warnings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app = _app(monkeypatch, tmp_path)
    _keyed(app.text_input, "pronunciation-global-new-pattern").set_value("Qwen").run()
    _keyed(app.text_input, "pronunciation-global-new-replacement").set_value("cuen").run()
    _keyed(app.button, "pronunciation-add-global").click().run()
    rules = app.session_state.global_pronunciation_rules
    assert [(rule.pattern, rule.replacement) for rule in rules] == [("Qwen", "cuen")]
    dictionary = AppPaths(tmp_path / "Music").pronunciation_dictionary
    assert dictionary.is_file()

    rule = rules[0]
    _keyed(app.checkbox, f"pronunciation-rule-enabled-{rule.rule_id}").uncheck().run()
    save = _keyed(app.button, f"pronunciation-rule-save-{rule.rule_id}")
    assert not save.disabled
    save.click().run()
    assert not app.session_state.global_pronunciation_rules[0].enabled
    assert not app.exception
    assert not any("session state" in warning.value.lower() for warning in app.warning)


def test_utterance_manual_override_local_rule_and_pending_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app = _app(monkeypatch, tmp_path)
    utterance = app.session_state.project.utterances[0]
    _keyed(app.text_area, f"utterance-text-{utterance.utterance_id}").set_value(
        "Qwen usa MOFA2"
    ).run()
    _keyed(
        app.checkbox,
        f"utterance-pronunciation-manual-{utterance.utterance_id}",
    ).check().run()
    _keyed(
        app.text_area,
        f"utterance-pronunciation-manual-text-{utterance.utterance_id}",
    ).set_value("cuen usa mofa dos").run()
    _keyed(
        app.button,
        f"utterance-pronunciation-apply-{utterance.utterance_id}",
    ).click().run()
    assert utterance.text == "Qwen usa MOFA2"
    assert utterance.manual_spoken_text_override == "cuen usa mofa dos"

    _keyed(
        app.button,
        f"utterance-pronunciation-reset-{utterance.utterance_id}",
    ).click().run()
    _keyed(
        app.button,
        f"utterance-pronunciation-pending-{utterance.utterance_id}",
    ).click().run()
    assert app.session_state.pronunciation_pending_terms
    assert not app.session_state.project.pronunciation_rules

    _keyed(
        app.text_input,
        f"utterance-pronunciation-term-{utterance.utterance_id}",
    ).set_value("Qwen").run()
    _keyed(
        app.text_input,
        f"utterance-pronunciation-spoken-{utterance.utterance_id}",
    ).set_value("cuen").run()
    _keyed(
        app.button,
        f"utterance-pronunciation-add-local-{utterance.utterance_id}",
    ).click().run()
    assert utterance.utterance_rules[0].scope == "utterance"
    assert utterance.utterance_rules[0].replacement == "cuen"
    assert not app.exception


def test_profile_change_marks_ready_audio_stale(monkeypatch, tmp_path: Path) -> None:
    app = _app(monkeypatch, tmp_path)
    project = app.session_state.project
    utterance = project.utterances[0]
    utterance.status = "ready"
    utterance.audio_relative_path = f"audio/normalized/{utterance.utterance_id}.wav"
    app.run()
    _keyed(
        app.selectbox,
        f"pronunciation-project-math-{project.project_id}",
    ).select("explicit").run()
    assert project.pronunciation_profile.math_style == "explicit"
    assert utterance.status == "stale"
    assert utterance.audio_relative_path is not None
    assert not app.exception
