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
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .audio import probe_audio
from .paths import AppPaths
from .qwen_gpu_safety import GpuPreflightResult, GpuSafetyPolicy, load_gpu_safety_policy
from .qwen_preflight import run_gpu_preflight
from .qwen_service import DEFAULT_MODEL
from .synthesis import SynthesisBusyError


class QwenClientError(RuntimeError):
    def __init__(self, message: str, *, code: str = "client_error", retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


QwenPythonSource = Literal["environment", "configured", "sibling-discovery"]


@dataclass(frozen=True)
class QwenPythonResolution:
    path: Path
    source: QwenPythonSource
    diagnostic: str


def _probe_qwen_python(
    path: Path,
    source: QwenPythonSource,
    *,
    runner: Any,
    timeout: float,
) -> QwenPythonResolution:
    if not path.exists():
        raise QwenClientError(
            f"No se encontró el runtime Qwen de origen {source}: {path}",
            code="qwen_runtime_not_found",
        )
    if not path.is_file():
        raise QwenClientError(
            f"La ruta del runtime Qwen de origen {source} no es un archivo: {path}",
            code="qwen_runtime_not_file",
        )
    if not os.access(path, os.X_OK):
        raise QwenClientError(
            f"El Python Qwen de origen {source} no es ejecutable: {path}",
            code="qwen_runtime_not_executable",
        )

    probes = (
        (
            "python",
            "import sys; print('qwen-runtime-python-ok')",
            "qwen-runtime-python-ok",
        ),
        (
            "qwen_tts",
            "import qwen_tts; print('qwen-tts-import-ok')",
            "qwen-tts-import-ok",
        ),
    )
    probe_environment = os.environ.copy()
    probe_environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    for probe_name, script, marker in probes:
        try:
            result = runner(
                [str(path), "-c", script],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=probe_environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise QwenClientError(
                f"El runtime Qwen agotó el tiempo al validar {probe_name}: {path}",
                code="qwen_runtime_probe_timeout",
            ) from exc
        except OSError as exc:
            raise QwenClientError(
                f"No se pudo ejecutar el Python Qwen de origen {source}: {path}: {exc}",
                code="qwen_runtime_not_python",
            ) from exc
        stdout = str(result.stdout or "")
        if result.returncode == 0 and marker in stdout.splitlines():
            continue
        detail = str(result.stderr or result.stdout or f"código {result.returncode}").strip()
        detail = detail[-500:] if detail else f"código {result.returncode}"
        if probe_name == "qwen_tts":
            raise QwenClientError(
                f"El Python Qwen no puede importar qwen_tts: {path}: {detail}",
                code="qwen_runtime_missing_qwen_tts",
            )
        raise QwenClientError(
            f"La ruta seleccionada no puede ejecutar Python correctamente: {path}: {detail}",
            code="qwen_runtime_not_python",
        )
    return QwenPythonResolution(
        path=path,
        source=source,
        diagnostic=f"Runtime Qwen válido ({source}); Python y qwen_tts comprobados sin síntesis.",
    )


@lru_cache(maxsize=16)
def _probe_qwen_python_cached(
    path: Path,
    source: QwenPythonSource,
    timeout: float,
    _file_signature: tuple[int, int, int],
) -> QwenPythonResolution:
    return _probe_qwen_python(path, source, runner=subprocess.run, timeout=timeout)


def resolve_qwen_python(
    configured_python: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    repository_root: Path | None = None,
    runner: Any | None = None,
    timeout: float = 30,
) -> QwenPythonResolution:
    """Resolve and validate the external Qwen interpreter without loading a model."""
    values = os.environ if env is None else env
    if "QWEN_TTS_PYTHON" in values:
        raw_path = values["QWEN_TTS_PYTHON"].strip()
        if not raw_path:
            raise QwenClientError(
                "QWEN_TTS_PYTHON está definida pero vacía",
                code="qwen_runtime_not_found",
            )
        candidate = Path(os.path.abspath(Path(raw_path).expanduser()))
        source: QwenPythonSource = "environment"
    elif configured_python is not None:
        raw_path = str(configured_python).strip()
        if not raw_path:
            raise QwenClientError(
                "La ruta Python Qwen configurada está vacía",
                code="qwen_runtime_not_found",
            )
        candidate = Path(os.path.abspath(Path(raw_path).expanduser()))
        source = "configured"
    else:
        repository = (
            repository_root.resolve(strict=False)
            if repository_root is not None
            else Path(__file__).resolve().parents[1]
        )
        candidate = Path(
            os.path.abspath(repository.parent / "qwen" / ".venv-qwen" / "bin" / "python")
        )
        source = "sibling-discovery"
    if runner is not None:
        return _probe_qwen_python(candidate, source, runner=runner, timeout=timeout)
    try:
        metadata = candidate.stat()
    except OSError:
        return _probe_qwen_python(candidate, source, runner=subprocess.run, timeout=timeout)
    signature = (metadata.st_mode, metadata.st_size, metadata.st_mtime_ns)
    return _probe_qwen_python_cached(candidate, source, timeout, signature)


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
    python_source: QwenPythonSource = "configured"
    python_diagnostic: str = "Runtime Qwen proporcionado mediante configuración explícita."

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        configured_python: str | Path | None = None,
    ) -> QwenClientConfig:
        values = os.environ if env is None else env
        runtime = resolve_qwen_python(configured_python, env=values)
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
            python=runtime.path,
            model=values.get("QWEN_TTS_MODEL", DEFAULT_MODEL),
            host="127.0.0.1",
            port=port,
            device=device,
            dtype=dtype,
            attention=attention,
            timeout=float(values.get("QWEN_TTS_TIMEOUT", "600")),
            python_source=runtime.source,
            python_diagnostic=runtime.diagnostic,
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

    def preflight(self) -> dict[str, Any]:
        return self._request("GET", "/preflight", timeout=30)

    def diagnostic(self) -> dict[str, Any]:
        return self._request("GET", "/diagnostic", timeout=30)

    def synthesize(
        self,
        *,
        text: str,
        speaker: str,
        language: str,
        generation_options: dict[str, int | float],
        output_path: Path,
        instruct: str | None = None,
        execution_mode: str = "cuda",
        confirm_cpu_fallback: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, object] = {
            "text": text,
            "speaker": speaker,
            "language": language,
            "generation_options": generation_options,
            "output_path": str(output_path),
            "execution_mode": execution_mode,
            "confirm_cpu_fallback": confirm_cpu_fallback,
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

    def stop_worker(self) -> dict[str, Any]:
        return self._request("POST", "/worker/stop", {})

    def cancel(self, *, stop_current: bool = True) -> dict[str, Any]:
        return self._request("POST", "/cancel", {"stop_current": stop_current})


def synthesize_qwen_text(
    voice_id: str,
    text: str,
    language: str,
    generation_options: dict[str, int | float],
    output_path: Path,
    *,
    client: QwenClient | None = None,
    execution_mode: str = "cuda",
    confirm_cpu_fallback: bool = False,
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
            execution_mode=execution_mode,
            confirm_cpu_fallback=confirm_cpu_fallback,
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
        policy: GpuSafetyPolicy | None = None,
    ) -> None:
        self.paths = paths
        self.config = config or QwenClientConfig.from_env()
        self.client = QwenClient(self.config)
        self._popen = popen
        self.policy = policy

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

    def gpu_policy(self) -> GpuSafetyPolicy:
        return self.policy or load_gpu_safety_policy(self.paths.qwen_gpu_policy)

    def preflight(self) -> GpuPreflightResult:
        return run_gpu_preflight(self.gpu_policy(), self.config.python, service_state="offline")

    def start(
        self,
        *,
        wait_seconds: float = 30,
        execution_mode: str = "cuda",
        confirm_cpu_fallback: bool = False,
    ) -> dict[str, Any]:
        current = self.status()
        if current.get("state") != "offline":
            return current
        if not self.config.python.is_file():
            raise QwenClientError(f"No existe el Python Qwen: {self.config.python}")
        if execution_mode == "cpu":
            if not (self.gpu_policy().allow_cpu_fallback and confirm_cpu_fallback):
                raise QwenClientError(
                    "El modo CPU requiere habilitación y confirmación explícitas",
                    code="cpu_fallback_not_confirmed",
                )
        elif execution_mode == "cuda":
            preflight = self.preflight()
            if not preflight.allowed:
                detail = preflight.blockers[0] if preflight.blockers else "estado GPU no seguro"
                raise QwenClientError(
                    "Generación bloqueada para proteger la sesión gráfica. " + detail,
                    code="gpu_preflight_blocked",
                )
        else:
            raise QwenClientError("Modo de ejecución Qwen desconocido", code="invalid_mode")
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
                    "QWEN_GPU_SAFETY_POLICY": json.dumps(
                        self.gpu_policy().to_dict(), separators=(",", ":")
                    ),
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
    parser.add_argument("command", choices=("runtime", "status", "start", "unload", "stop"))
    arguments = parser.parse_args()
    try:
        config = QwenClientConfig.from_env()
        if arguments.command == "runtime":
            result = {
                "ok": True,
                "path": str(config.python),
                "source": config.python_source,
                "diagnostic": config.python_diagnostic,
            }
        else:
            manager = QwenBackendManager(AppPaths.discover(), config)
            if arguments.command == "start":
                result = manager.start()
            elif arguments.command == "unload":
                result = manager.client.unload()
            elif arguments.command == "stop":
                result = manager.stop()
            else:
                result = manager.status()
    except (OSError, ValueError, QwenClientError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "code": getattr(exc, "code", "configuration_error"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
