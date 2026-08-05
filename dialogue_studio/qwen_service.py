"""Isolated localhost service for Qwen3-TTS.

Torch and qwen_tts are imported lazily inside the dedicated Qwen process, never by the
main Streamlit application.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
FALLBACK_SPEAKERS = (
    "aiden",
    "dylan",
    "eric",
    "ono_anna",
    "ryan",
    "serena",
    "sohee",
    "uncle_fu",
    "vivian",
)
FALLBACK_LANGUAGES = (
    "auto",
    "chinese",
    "english",
    "french",
    "german",
    "italian",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "spanish",
)
GENERATION_OPTION_LIMITS: dict[str, tuple[type, float, float]] = {
    "seed": (int, 0, 4_294_967_295),
    "max_new_tokens": (int, 64, 8192),
    "temperature": (float, 0.1, 2.0),
    "top_p": (float, 0.1, 1.0),
    "top_k": (int, 1, 200),
    "repetition_penalty": (float, 0.8, 2.0),
}
DEFAULT_GENERATION_OPTIONS: dict[str, int | float] = {
    "seed": 0,
    "max_new_tokens": 8192,
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.05,
}


class QwenServiceError(RuntimeError):
    """Structured service error safe to return to a localhost client."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = HTTPStatus.BAD_REQUEST,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = int(status)
        self.retryable = retryable


@dataclass(frozen=True)
class QwenServiceConfig:
    model: str
    host: str
    port: int
    device: str
    dtype: str
    attention: str
    timeout: float
    output_root: Path
    pid_file: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> QwenServiceConfig:
        values = os.environ if env is None else env
        host = values.get("QWEN_TTS_HOST", "127.0.0.1")
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("QWEN_TTS_HOST debe ser 127.0.0.1")
        port = int(values.get("QWEN_TTS_PORT", "8765"))
        if not 1 <= port <= 65_535:
            raise ValueError("QWEN_TTS_PORT debe estar entre 1 y 65535")
        device = values.get("QWEN_TTS_DEVICE", "cuda:0")
        dtype = values.get("QWEN_TTS_DTYPE", "bfloat16")
        attention = values.get("QWEN_TTS_ATTN", "sdpa")
        if device != "cuda:0" or dtype != "bfloat16" or attention != "sdpa":
            raise ValueError("Esta integración requiere cuda:0, bfloat16 y sdpa")
        output_root = Path(
            values.get(
                "QWEN_TTS_OUTPUT_ROOT",
                str(Path.home() / "Música" / "SpeechNote Dialogue Studio"),
            )
        ).expanduser()
        runtime_dir = Path(
            values.get(
                "QWEN_TTS_RUNTIME_DIR",
                str(Path.home() / ".cache" / "speechnote-dialogue-studio"),
            )
        ).expanduser()
        return cls(
            model=values.get("QWEN_TTS_MODEL", DEFAULT_MODEL),
            host="127.0.0.1",
            port=port,
            device=device,
            dtype=dtype,
            attention=attention,
            timeout=float(values.get("QWEN_TTS_TIMEOUT", "600")),
            output_root=output_root,
            pid_file=runtime_dir / f"qwen-tts-{port}.pid",
        )


def validate_generation_options(value: object) -> dict[str, int | float]:
    if value is None:
        return dict(DEFAULT_GENERATION_OPTIONS)
    if not isinstance(value, dict):
        raise QwenServiceError("invalid_options", "generation_options debe ser un objeto JSON")
    unknown = set(value) - set(GENERATION_OPTION_LIMITS)
    if unknown:
        raise QwenServiceError(
            "unsupported_option",
            f"Opciones de generación no soportadas: {', '.join(sorted(unknown))}",
        )
    validated = dict(DEFAULT_GENERATION_OPTIONS)
    for name, raw in value.items():
        expected, minimum, maximum = GENERATION_OPTION_LIMITS[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise QwenServiceError("invalid_option", f"{name} debe ser numérico")
        if expected is int and not isinstance(raw, int):
            raise QwenServiceError("invalid_option", f"{name} debe ser entero")
        number = expected(raw)
        if not minimum <= number <= maximum:
            raise QwenServiceError(
                "invalid_option", f"{name} debe estar entre {minimum:g} y {maximum:g}"
            )
        validated[name] = number
    return validated


def _safe_output_path(root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise QwenServiceError("invalid_output", "output_path es obligatorio")
    candidate = Path(raw_path)
    if not candidate.is_absolute() or candidate.suffix.lower() != ".wav":
        raise QwenServiceError("invalid_output", "La salida debe ser una ruta WAV absoluta")
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve(strict=False)
    if root_resolved not in candidate_resolved.parents:
        raise QwenServiceError("unsafe_output", "La salida está fuera de la carpeta controlada")
    cursor = candidate.absolute()
    lexical_root = root.absolute()
    while cursor != lexical_root:
        if cursor.is_symlink():
            raise QwenServiceError("unsafe_output", "No se permiten enlaces simbólicos")
        cursor = cursor.parent
    return candidate_resolved


class QwenRuntime:
    """Owns one lazy model instance and serializes GPU generation."""

    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        model_loader: Callable[[QwenServiceConfig, Any], Any] | None = None,
        torch_loader: Callable[[], Any] | None = None,
        wave_writer: Callable[..., None] | None = None,
        finite_checker: Callable[[Any], bool] | None = None,
    ) -> None:
        self.config = config
        self._model_loader = model_loader or self._default_model_loader
        self._torch_loader = torch_loader or self._default_torch_loader
        self._wave_writer = wave_writer or self._default_wave_writer
        self._finite_checker = finite_checker or self._default_finite_checker
        self._state_lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._model: Any | None = None
        self._torch: Any | None = None
        self._state = "starting"
        self._last_error: str | None = None
        self._speakers = list(FALLBACK_SPEAKERS)
        self._languages = list(FALLBACK_LANGUAGES)
        self._supports_instruct = False
        self._load_count = 0

    @staticmethod
    def _default_torch_loader() -> Any:
        import torch

        return torch

    @staticmethod
    def _default_model_loader(config: QwenServiceConfig, torch_module: Any) -> Any:
        from qwen_tts import Qwen3TTSModel

        return Qwen3TTSModel.from_pretrained(
            config.model,
            device_map=config.device,
            dtype=torch_module.bfloat16,
            attn_implementation=config.attention,
        )

    @staticmethod
    def _default_wave_writer(path: Path, waveform: Any, sample_rate: int) -> None:
        import soundfile as sf

        sf.write(path, waveform, sample_rate, subtype="PCM_16", format="WAV")

    @staticmethod
    def _default_finite_checker(waveform: Any) -> bool:
        import numpy as np

        return bool(np.isfinite(waveform).all())

    def initialize(self) -> None:
        try:
            torch_module = self._get_torch()
            if not torch_module.cuda.is_available():
                raise RuntimeError("CUDA no está disponible")
            if not torch_module.cuda.is_bf16_supported():
                raise RuntimeError("La GPU no soporta bfloat16")
        except Exception as exc:
            self._set_state("error", str(exc))
            raise
        self._set_state("idle", None)

    def _get_torch(self) -> Any:
        if self._torch is None:
            self._torch = self._torch_loader()
        return self._torch

    def _set_state(self, state: str, error: str | None = None) -> None:
        with self._state_lock:
            self._state = state
            self._last_error = error

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        self._set_state("loading_model")
        try:
            model = self._model_loader(self.config, self._get_torch())
            speakers = [str(value).lower() for value in model.get_supported_speakers()]
            languages = [str(value).lower() for value in model.get_supported_languages()]
            if speakers:
                self._speakers = speakers
            if languages:
                self._languages = languages
            size = str(getattr(getattr(model, "model", None), "tts_model_size", ""))
            self._supports_instruct = size not in "0b6"
            self._model = model
            self._load_count += 1
            return model
        except Exception as exc:
            self._set_state("error", str(exc))
            raise

    def _gpu_details(self) -> dict[str, object]:
        torch_module = self._get_torch()
        cuda = torch_module.cuda
        available = bool(cuda.is_available())
        free: int | None = None
        total: int | None = None
        if available:
            with contextlib.suppress(AttributeError, RuntimeError):
                free, total = (int(value) for value in cuda.mem_get_info(0))
        return {
            "torch": str(torch_module.__version__),
            "cuda_available": available,
            "cuda_version": str(getattr(getattr(torch_module, "version", None), "cuda", "")),
            "bf16": bool(available and cuda.is_bf16_supported()),
            "gpu": str(cuda.get_device_name(0)) if available else None,
            "vram_free_bytes": free,
            "vram_total_bytes": total,
        }

    def health(self) -> dict[str, object]:
        with self._state_lock:
            state = self._state
            error = self._last_error
        return {
            "ok": state != "error",
            "service": "speechnote-dialogue-studio-qwen",
            "state": state,
            "model": self.config.model,
            "model_loaded": self._model is not None,
            "load_count": self._load_count,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "attention": self.config.attention,
            "native_sample_rate": 24_000,
            "last_error": error,
            **self._gpu_details(),
        }

    def capabilities(self) -> dict[str, object]:
        return {
            "ok": True,
            "speakers": list(self._speakers),
            "languages": list(self._languages),
            "source": "model" if self._model is not None else "verified_fallback",
            "supports_instruct": self._supports_instruct,
            "supports_voice_design": False,
            "supports_voice_cloning": False,
            "supports_sampling_controls": True,
            "supports_speaker_selection": True,
            "supports_language_selection": True,
            "generation_options": {
                name: {"type": expected.__name__, "min": minimum, "max": maximum}
                for name, (expected, minimum, maximum) in GENERATION_OPTION_LIMITS.items()
            },
            "defaults": dict(DEFAULT_GENERATION_OPTIONS),
        }

    def synthesize(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise QwenServiceError("invalid_request", "El cuerpo debe ser un objeto JSON")
        text = payload.get("text")
        speaker = payload.get("speaker")
        language = payload.get("language", "auto")
        instruct = payload.get("instruct")
        if not isinstance(text, str) or not text.strip():
            raise QwenServiceError("invalid_text", "text no puede estar vacío")
        if len(text) > 20_000:
            raise QwenServiceError("invalid_text", "text excede 20000 caracteres")
        if not isinstance(speaker, str) or speaker.lower() not in self._speakers:
            raise QwenServiceError("invalid_speaker", "La voz Qwen no está soportada")
        if not isinstance(language, str) or language.lower() not in self._languages:
            raise QwenServiceError("invalid_language", "El idioma Qwen no está soportado")
        if instruct is not None and not isinstance(instruct, str):
            raise QwenServiceError("invalid_instruct", "instruct debe ser texto")
        destination = _safe_output_path(self.config.output_root, payload.get("output_path"))
        options = validate_generation_options(payload.get("generation_options"))
        if not self._generation_lock.acquire(blocking=False):
            raise QwenServiceError(
                "gpu_busy",
                "Ya hay una síntesis Qwen realmente activa",
                status=HTTPStatus.CONFLICT,
                retryable=True,
            )
        partial: Path | None = None
        started = time.monotonic()
        try:
            model = self._ensure_model()
            if instruct and not self._supports_instruct:
                self._set_state("idle", None)
                raise QwenServiceError(
                    "unsupported_instruct",
                    "El modelo Qwen 0.6B instalado no admite instruct",
                )
            self._set_state("generating")
            torch_module = self._get_torch()
            seed = int(options.pop("seed"))
            devices = [0] if torch_module.cuda.is_available() else []
            with torch_module.random.fork_rng(devices=devices, enabled=True):
                torch_module.manual_seed(seed)
                if devices:
                    torch_module.cuda.manual_seed_all(seed)
                call: dict[str, object] = {
                    "text": text.strip(),
                    "speaker": speaker.lower(),
                    "language": language.lower(),
                    "non_streaming_mode": True,
                    **options,
                }
                if self._supports_instruct and instruct:
                    call["instruct"] = instruct.strip()
                waveforms, sample_rate = model.generate_custom_voice(**call)
            if not waveforms:
                raise RuntimeError("Qwen no devolvió audio")
            waveform = waveforms[0]
            if not self._finite_checker(waveform):
                raise RuntimeError("Qwen produjo muestras NaN o infinitas")
            sample_rate = int(sample_rate)
            if sample_rate <= 0:
                raise RuntimeError("Qwen devolvió una frecuencia inválida")
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            partial = destination.parent / f".{destination.name}.{uuid4().hex}.partial"
            self._wave_writer(partial, waveform, sample_rate)
            with partial.open("rb") as handle:
                header = handle.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise RuntimeError("La salida temporal Qwen no es RIFF/WAVE")
            if destination.is_symlink():
                raise RuntimeError("No se sobrescriben enlaces simbólicos")
            os.replace(partial, destination)
            partial = None
            self._set_state("idle", None)
            return {
                "ok": True,
                "output_path": str(destination),
                "sample_rate": sample_rate,
                "duration_seconds": len(waveform) / sample_rate,
                "elapsed_seconds": time.monotonic() - started,
                "speaker": speaker.lower(),
                "language": language.lower(),
            }
        except QwenServiceError:
            raise
        except Exception as exc:
            self._set_state("error", str(exc))
            raise QwenServiceError(
                "generation_failed", str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR
            ) from exc
        finally:
            self._generation_lock.release()

    def unload(self) -> dict[str, object]:
        if not self._generation_lock.acquire(blocking=False):
            raise QwenServiceError(
                "gpu_busy",
                "No se puede descargar el modelo durante una síntesis",
                status=HTTPStatus.CONFLICT,
                retryable=True,
            )
        try:
            was_loaded = self._model is not None
            self._model = None
            if self._torch is not None:
                import gc

                gc.collect()
                self._torch.cuda.empty_cache()
            self._set_state("idle", None)
            return {"ok": True, "unloaded": was_loaded}
        finally:
            self._generation_lock.release()


class QwenHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], runtime: QwenRuntime):
        super().__init__(address, QwenRequestHandler)
        self.runtime = runtime


class QwenRequestHandler(BaseHTTPRequestHandler):
    server: QwenHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, status: int, data: dict[str, object]) -> None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _payload(self) -> object:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise QwenServiceError("invalid_request", "Content-Length inválido") from exc
        if length <= 0 or length > 1_000_000:
            raise QwenServiceError("invalid_request", "Tamaño de petición inválido")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QwenServiceError("invalid_json", "El cuerpo JSON no es válido") from exc

    def _error(self, error: QwenServiceError) -> None:
        self._send(
            error.status,
            {
                "ok": False,
                "state": self.server.runtime.health()["state"],
                "error": {
                    "code": error.code,
                    "message": str(error),
                    "retryable": error.retryable,
                },
            },
        )

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path == "/health":
                self._send(HTTPStatus.OK, self.server.runtime.health())
            elif self.path == "/capabilities":
                self._send(HTTPStatus.OK, self.server.runtime.capabilities())
            else:
                raise QwenServiceError("not_found", "Endpoint desconocido", status=404)
        except QwenServiceError as exc:
            self._error(exc)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/synthesize":
                self._send(HTTPStatus.OK, self.server.runtime.synthesize(self._payload()))
            elif self.path == "/unload":
                self._send(HTTPStatus.OK, self.server.runtime.unload())
            elif self.path == "/shutdown":
                self.server.runtime.unload()
                self._send(HTTPStatus.OK, {"ok": True, "state": "offline"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                raise QwenServiceError("not_found", "Endpoint desconocido", status=404)
        except QwenServiceError as exc:
            self._error(exc)
        except Exception as exc:
            self._error(
                QwenServiceError(
                    "internal_error", str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR
                )
            )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _pid_is_qwen_service(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ")
    except OSError:
        return False
    return b"dialogue_studio.qwen_service" in command


def _claim_pid_file(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            existing = -1
        if _pid_alive(existing) and _pid_is_qwen_service(existing):
            raise RuntimeError(f"Ya existe un backend Qwen activo con PID {existing}")
        path.unlink(missing_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()))
        handle.flush()
        os.fsync(handle.fileno())


def _release_pid_file(path: Path) -> None:
    try:
        owner = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return
    if owner == os.getpid():
        path.unlink(missing_ok=True)


def serve(config: QwenServiceConfig) -> None:
    _claim_pid_file(config.pid_file)
    server: QwenHTTPServer | None = None
    try:
        runtime = QwenRuntime(config)
        runtime.initialize()
        server = QwenHTTPServer((config.host, config.port), runtime)

        def stop(_signum: int, _frame: object) -> None:
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve_forever(poll_interval=0.25)
        runtime.unload()
    finally:
        if server is not None:
            server.server_close()
        _release_pid_file(config.pid_file)


def main() -> None:
    argparse.ArgumentParser(description="Servicio local aislado Qwen3-TTS").parse_args()
    serve(QwenServiceConfig.from_env())


if __name__ == "__main__":
    main()
