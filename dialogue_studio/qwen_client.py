"""Standard-library client and lifecycle manager for the isolated Qwen service."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import probe_audio
from .paths import AppPaths
from .qwen_service import DEFAULT_MODEL
from .synthesis import SynthesisBusyError

DEFAULT_QWEN_PYTHON = Path("/home/enriquedo/PersonalProjects/qwen/.venv-qwen/bin/python")


class QwenClientError(RuntimeError):
    def __init__(self, message: str, *, code: str = "client_error", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class QwenClientConfig:
    python: Path
    model: str
    host: str
    port: int
    device: str
    dtype: str
    attention: str
    timeout: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> QwenClientConfig:
        values = os.environ if env is None else env
        host = values.get("QWEN_TTS_HOST", "127.0.0.1")
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("QWEN_TTS_HOST debe ser local")
        port = int(values.get("QWEN_TTS_PORT", "8765"))
        if not 1 <= port <= 65_535:
            raise ValueError("QWEN_TTS_PORT debe estar entre 1 y 65535")
        device = values.get("QWEN_TTS_DEVICE", "cuda:0")
        dtype = values.get("QWEN_TTS_DTYPE", "bfloat16")
        attention = values.get("QWEN_TTS_ATTN", "sdpa")
        if device != "cuda:0" or dtype != "bfloat16" or attention != "sdpa":
            raise ValueError("Esta integración requiere cuda:0, bfloat16 y sdpa")
        return cls(
            python=Path(values.get("QWEN_TTS_PYTHON", str(DEFAULT_QWEN_PYTHON))).expanduser(),
            model=values.get("QWEN_TTS_MODEL", DEFAULT_MODEL),
            host="127.0.0.1",
            port=port,
            device=device,
            dtype=dtype,
            attention=attention,
            timeout=float(values.get("QWEN_TTS_TIMEOUT", "600")),
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class QwenClient:
    def __init__(self, config: QwenClientConfig | None = None):
        self.config = config or QwenClientConfig.from_env()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body = None
        headers: dict[str, str] = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.config.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout if timeout is not None else 5
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                data = json.loads(exc.read().decode("utf-8"))
                details = data.get("error", {})
                raise QwenClientError(
                    str(details.get("message", f"Error HTTP {exc.code}")),
                    code=str(details.get("code", "http_error")),
                    retryable=bool(details.get("retryable", False)),
                ) from exc
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                raise QwenClientError(f"El backend Qwen devolvió HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise QwenClientError(
                "El backend Qwen no está disponible", code="offline", retryable=True
            ) from exc
        except json.JSONDecodeError as exc:
            raise QwenClientError("El backend Qwen devolvió JSON inválido") from exc
        if not isinstance(data, dict):
            raise QwenClientError("Respuesta Qwen inválida")
        return data

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def capabilities(self) -> dict[str, Any]:
        return self._request("GET", "/capabilities")

    def synthesize(
        self,
        *,
        text: str,
        speaker: str,
        language: str,
        generation_options: dict[str, int | float],
        output_path: Path,
        instruct: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, object] = {
            "text": text,
            "speaker": speaker,
            "language": language,
            "generation_options": generation_options,
            "output_path": str(output_path),
        }
        if instruct:
            payload["instruct"] = instruct
        return self._request(
            "POST", "/synthesize", payload, timeout=self.config.timeout
        )

    def unload(self) -> dict[str, Any]:
        return self._request("POST", "/unload", {})

    def shutdown(self) -> dict[str, Any]:
        return self._request("POST", "/shutdown", {})


def synthesize_qwen_text(
    voice_id: str,
    text: str,
    language: str,
    generation_options: dict[str, int | float],
    output_path: Path,
    *,
    client: QwenClient | None = None,
) -> dict[str, Any]:
    """Generate and validate one native 24 kHz Qwen WAV without importing Qwen."""
    active_client = client or QwenClient()
    try:
        response = active_client.synthesize(
            text=text,
            speaker=voice_id,
            language=language,
            generation_options=generation_options,
            output_path=output_path,
        )
    except QwenClientError as exc:
        if exc.code == "gpu_busy":
            raise SynthesisBusyError("Hay una síntesis Qwen real activa") from exc
        raise
    try:
        reported_path = Path(str(response["output_path"])).resolve()
    except (KeyError, OSError, ValueError) as exc:
        raise QwenClientError("Qwen no confirmó una ruta de salida válida") from exc
    if reported_path != output_path.resolve():
        raise QwenClientError("Qwen confirmó una ruta de salida inesperada")
    info = probe_audio(output_path)
    if (
        info.codec != "pcm_s16le"
        or info.sample_rate != 24_000
        or info.channels != 1
        or info.duration_seconds <= 0
    ):
        raise QwenClientError("Qwen no produjo PCM 16-bit mono a 24000 Hz")
    return response


class QwenBackendManager:
    def __init__(
        self,
        paths: AppPaths,
        config: QwenClientConfig | None = None,
        *,
        popen: Any = subprocess.Popen,
    ) -> None:
        self.paths = paths
        self.config = config or QwenClientConfig.from_env()
        self.client = QwenClient(self.config)
        self._popen = popen

    @property
    def runtime_dir(self) -> Path:
        return self.paths.root / "runtime"

    @property
    def pid_file(self) -> Path:
        return self.runtime_dir / f"qwen-tts-{self.config.port}.pid"

    @property
    def log_file(self) -> Path:
        return self.runtime_dir / "qwen-tts.log"

    def _read_pid(self) -> int | None:
        try:
            return int(self.pid_file.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        return True

    @staticmethod
    def _pid_is_qwen_service(pid: int) -> bool:
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ")
        except OSError:
            return False
        return b"dialogue_studio.qwen_service" in command

    def clean_stale_pid(self) -> bool:
        pid = self._read_pid()
        if pid is None:
            if self.pid_file.exists():
                self.pid_file.unlink(missing_ok=True)
                return True
            return False
        if self._pid_alive(pid) and self._pid_is_qwen_service(pid):
            return False
        self.pid_file.unlink(missing_ok=True)
        return True

    def status(self) -> dict[str, Any]:
        try:
            health = self.client.health()
            if health.get("service") != "speechnote-dialogue-studio-qwen":
                raise QwenClientError("El puerto Qwen está ocupado por otro servicio")
            return health
        except QwenClientError as exc:
            self.clean_stale_pid()
            return {
                "ok": False,
                "state": "offline",
                "model": self.config.model,
                "last_error": str(exc) if exc.code != "offline" else None,
            }

    def start(self, *, wait_seconds: float = 30) -> dict[str, Any]:
        current = self.status()
        if current.get("state") != "offline":
            return current
        if not self.config.python.is_file():
            raise QwenClientError(f"No existe el Python Qwen: {self.config.python}")
        self.runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.runtime_dir / "qwen-start.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            current = self.status()
            if current.get("state") != "offline":
                return current
            self.clean_stale_pid()
            environment = os.environ.copy()
            environment.update(
                {
                    "QWEN_TTS_MODEL": self.config.model,
                    "QWEN_TTS_HOST": self.config.host,
                    "QWEN_TTS_PORT": str(self.config.port),
                    "QWEN_TTS_DEVICE": self.config.device,
                    "QWEN_TTS_DTYPE": self.config.dtype,
                    "QWEN_TTS_ATTN": self.config.attention,
                    "QWEN_TTS_TIMEOUT": str(self.config.timeout),
                    "QWEN_TTS_OUTPUT_ROOT": str(self.paths.root),
                    "QWEN_TTS_RUNTIME_DIR": str(self.runtime_dir),
                }
            )
            repository = Path(__file__).resolve().parents[1]
            with self.log_file.open("ab", buffering=0) as log:
                try:
                    self._popen(
                        [str(self.config.python), "-m", "dialogue_studio.qwen_service"],
                        cwd=repository,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                        close_fds=True,
                    )
                except Exception:
                    lock_path.unlink(missing_ok=True)
                    raise
        lock_path.unlink(missing_ok=True)
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            status = self.status()
            if status.get("state") != "offline":
                return status
            time.sleep(0.2)
        raise QwenClientError(
            f"El backend Qwen no inició; revisa {self.log_file}",
            code="start_timeout",
            retryable=True,
        )

    def stop(self, *, wait_seconds: float = 10) -> dict[str, Any]:
        status = self.status()
        if status.get("state") == "offline":
            return status
        with contextlib.suppress(QwenClientError):
            self.client.shutdown()
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            status = self.status()
            if status.get("state") == "offline":
                return status
            time.sleep(0.2)
        pid = self._read_pid()
        if pid and self._pid_alive(pid) and self._pid_is_qwen_service(pid):
            os.kill(pid, signal.SIGTERM)
        return self.status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Control del backend local Qwen3-TTS")
    parser.add_argument("command", choices=("status", "start", "unload", "stop"))
    arguments = parser.parse_args()
    manager = QwenBackendManager(AppPaths.discover())
    if arguments.command == "start":
        result = manager.start()
    elif arguments.command == "unload":
        result = manager.client.unload()
    elif arguments.command == "stop":
        result = manager.stop()
    else:
        result = manager.status()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
