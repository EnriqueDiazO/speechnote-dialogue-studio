from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from dialogue_studio.qwen_client import QwenClient, QwenClientConfig, QwenClientError


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
