from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dialogue_studio.qwen_client import QwenClientError, resolve_qwen_python


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class SuccessfulProbe:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str]] = []

    def __call__(self, command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        environment = _kwargs.get("env")
        assert isinstance(environment, dict)
        self.environments.append(environment)
        marker = "qwen-tts-import-ok" if "qwen_tts" in command[-1] else "qwen-runtime-python-ok"
        return subprocess.CompletedProcess(command, 0, stdout=marker + "\n", stderr="")


def test_environment_runtime_has_highest_priority(tmp_path: Path) -> None:
    environment_python = _executable(tmp_path / "environment" / "python")
    configured_python = _executable(tmp_path / "configured" / "python")
    repository = tmp_path / "workspace" / "speechnote-dialogue-studio"
    _executable(repository.parent / "qwen" / ".venv-qwen" / "bin" / "python")
    probe = SuccessfulProbe()

    resolution = resolve_qwen_python(
        configured_python,
        env={"QWEN_TTS_PYTHON": str(environment_python)},
        repository_root=repository,
        runner=probe,
    )

    assert resolution.path == environment_python
    assert resolution.source == "environment"
    assert {command[0] for command in probe.commands} == {str(environment_python)}


def test_configured_runtime_precedes_sibling_discovery(tmp_path: Path) -> None:
    configured_python = _executable(tmp_path / "configured" / "python")
    repository = tmp_path / "workspace" / "speechnote-dialogue-studio"
    _executable(repository.parent / "qwen" / ".venv-qwen" / "bin" / "python")
    probe = SuccessfulProbe()

    resolution = resolve_qwen_python(
        configured_python,
        env={},
        repository_root=repository,
        runner=probe,
    )

    assert resolution.path == configured_python
    assert resolution.source == "configured"


def test_sibling_discovery_uses_temporary_repository_parent(tmp_path: Path) -> None:
    repository = tmp_path / "portable-workspace" / "dialogue-studio"
    actual_python = _executable(tmp_path / "interpreters" / "python")
    sibling_python = repository.parent / "qwen" / ".venv-qwen" / "bin" / "python"
    sibling_python.parent.mkdir(parents=True)
    sibling_python.symlink_to(actual_python)

    resolution = resolve_qwen_python(
        env={},
        repository_root=repository,
        runner=SuccessfulProbe(),
    )

    assert resolution.path == sibling_python
    assert resolution.source == "sibling-discovery"
    assert "sin síntesis" in resolution.diagnostic


def test_sibling_discovery_does_not_depend_on_a_user_name(tmp_path: Path) -> None:
    repository = tmp_path / "another-user" / "code" / "studio"
    sibling_python = _executable(
        repository.parent / "qwen" / ".venv-qwen" / "bin" / "python"
    )

    resolution = resolve_qwen_python(
        env={},
        repository_root=repository,
        runner=SuccessfulProbe(),
    )

    assert resolution.path == sibling_python
    assert resolution.source == "sibling-discovery"


def test_missing_runtime_is_a_controlled_error_and_creates_nothing(tmp_path: Path) -> None:
    repository = tmp_path / "workspace" / "studio"
    before = set(tmp_path.rglob("*"))

    with pytest.raises(QwenClientError, match="No se encontró") as error:
        resolve_qwen_python(env={}, repository_root=repository, runner=SuccessfulProbe())

    assert error.value.code == "qwen_runtime_not_found"
    assert set(tmp_path.rglob("*")) == before


def test_non_executable_runtime_is_rejected(tmp_path: Path) -> None:
    python = tmp_path / "qwen" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("not executable", encoding="utf-8")
    python.chmod(0o644)

    with pytest.raises(QwenClientError, match="no es ejecutable") as error:
        resolve_qwen_python(env={"QWEN_TTS_PYTHON": str(python)}, runner=SuccessfulProbe())

    assert error.value.code == "qwen_runtime_not_executable"


def test_python_without_qwen_tts_is_rejected_clearly(tmp_path: Path) -> None:
    python = _executable(tmp_path / "qwen" / "python")
    commands: list[list[str]] = []

    def probe(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "qwen_tts" in command[-1]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'qwen_tts'",
            )
        return subprocess.CompletedProcess(
            command, 0, stdout="qwen-runtime-python-ok\n", stderr=""
        )

    with pytest.raises(QwenClientError, match="no puede importar qwen_tts") as error:
        resolve_qwen_python(env={"QWEN_TTS_PYTHON": str(python)}, runner=probe)

    assert error.value.code == "qwen_runtime_missing_qwen_tts"
    assert len(commands) == 2


def test_runtime_validation_never_requests_a_model_download(tmp_path: Path) -> None:
    python = _executable(tmp_path / "qwen" / "python")
    probe = SuccessfulProbe()

    resolve_qwen_python(env={"QWEN_TTS_PYTHON": str(python)}, runner=probe)

    scripts = [command[-1] for command in probe.commands]
    assert scripts == [
        "import sys; print('qwen-runtime-python-ok')",
        "import qwen_tts; print('qwen-tts-import-ok')",
    ]
    assert all(
        "from_pretrained" not in script and "snapshot_download" not in script
        for script in scripts
    )
    assert all(environment["HF_HUB_OFFLINE"] == "1" for environment in probe.environments)
    assert all(environment["TRANSFORMERS_OFFLINE"] == "1" for environment in probe.environments)
