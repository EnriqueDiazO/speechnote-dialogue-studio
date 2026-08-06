from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from dialogue_studio.paths import AppPaths
from dialogue_studio.qwen_client import (
    QwenBackendManager,
    QwenClient,
    QwenClientConfig,
    QwenClientError,
)
from dialogue_studio.qwen_gpu_safety import GpuPreflightResult, GpuSafetyPolicy


def config(tmp_path: Path) -> QwenClientConfig:
    return QwenClientConfig(
        python=tmp_path / "python",
        model="model",
        host="127.0.0.1",
        port=8765,
        device="cuda:0",
        dtype="bfloat16",
        attention="sdpa",
        timeout=10,
    )


class Response:
    def __init__(self, data: dict[str, object]):
        self.data = json.dumps(data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self.data


def test_client_uses_local_json_endpoints_and_omits_empty_instruct(
    monkeypatch, tmp_path: Path
) -> None:
    requests = []

    def open_request(request, timeout):
        requests.append((request, timeout))
        return Response({"ok": True, "state": "idle"})

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    client = QwenClient(config(tmp_path))
    assert client.health()["state"] == "idle"
    client.synthesize(
        text="Hola",
        speaker="serena",
        language="spanish",
        generation_options={"seed": 1},
        output_path=tmp_path / "out.wav",
    )
    assert requests[0][0].full_url == "http://127.0.0.1:8765/health"
    payload = json.loads(requests[1][0].data)
    assert payload["speaker"] == "serena"
    assert "instruct" not in payload
    assert payload["execution_mode"] == "cuda"
    assert payload["confirm_cpu_fallback"] is False
    assert requests[1][1] == 10


def test_client_preserves_structured_http_error(monkeypatch, tmp_path: Path) -> None:
    body = io.BytesIO(
        json.dumps(
            {
                "error": {
                    "code": "gpu_busy",
                    "message": "GPU ocupada",
                    "retryable": True,
                }
            }
        ).encode()
    )

    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError("url", 409, "conflict", {}, body)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(QwenClientError) as error:
        QwenClient(config(tmp_path)).health()
    assert error.value.code == "gpu_busy"
    assert error.value.retryable is True


def test_client_reports_offline_without_leaking_transport_details(
    monkeypatch, tmp_path: Path
) -> None:
    def fail(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(QwenClientError, match="no está disponible") as error:
        QwenClient(config(tmp_path)).health()
    assert error.value.code == "offline"


def test_backend_start_removes_temporary_start_lock(monkeypatch, tmp_path: Path) -> None:
    settings = config(tmp_path)
    settings.python.write_text("fake", encoding="utf-8")
    manager = QwenBackendManager(
        AppPaths(tmp_path / "Music"),
        settings,
        popen=lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        manager,
        "preflight",
        lambda: GpuPreflightResult(allowed=True, timestamp="2026-08-05T00:00:00Z"),
    )
    states = iter(
        (
            {"state": "offline"},
            {"state": "offline"},
            {"state": "idle", "service": "speechnote-dialogue-studio-qwen"},
        )
    )
    monkeypatch.setattr(manager, "status", lambda: next(states))
    result = manager.start(wait_seconds=1)
    assert result["state"] == "idle"
    assert not (manager.runtime_dir / "qwen-start.lock").exists()


def test_backend_cpu_start_requires_confirmation_and_skips_gpu_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    settings = config(tmp_path)
    settings.python.write_text("fake", encoding="utf-8")
    calls = []
    manager = QwenBackendManager(
        AppPaths(tmp_path / "Music"),
        settings,
        popen=lambda *args, **kwargs: calls.append((args, kwargs)) or object(),
        policy=GpuSafetyPolicy(allow_cpu_fallback=True),
    )
    monkeypatch.setattr(
        manager,
        "preflight",
        lambda: pytest.fail("CPU start must not run the GPU preflight"),
    )
    monkeypatch.setattr(
        manager,
        "status",
        lambda: {"state": "offline"} if len(calls) == 0 else {"state": "idle"},
    )
    with pytest.raises(QwenClientError) as error:
        manager.start(execution_mode="cpu", confirm_cpu_fallback=False)
    assert error.value.code == "cpu_fallback_not_confirmed"
    result = manager.start(execution_mode="cpu", confirm_cpu_fallback=True)
    assert result["state"] == "idle"
    assert len(calls) == 1
