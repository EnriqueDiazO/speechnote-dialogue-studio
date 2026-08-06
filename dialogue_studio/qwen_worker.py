"""Single-job CUDA worker controlled through JSON lines on standard I/O."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from .qwen_service import (
    FALLBACK_LANGUAGES,
    FALLBACK_SPEAKERS,
    QwenServiceConfig,
    QwenServiceError,
    _safe_output_path,
    validate_generation_options,
)


class QwenRuntime:
    """Own one lazy CUDA model instance inside the disposable worker."""

    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        model_loader: Callable[[QwenServiceConfig, Any], Any] | None = None,
        torch_loader: Callable[[], Any] | None = None,
        wave_writer: Callable[..., None] | None = None,
        finite_checker: Callable[[Any], bool] | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._model_loader = model_loader or self._default_model_loader
        self._torch_loader = torch_loader or self._default_torch_loader
        self._wave_writer = wave_writer or self._default_wave_writer
        self._finite_checker = finite_checker or self._default_finite_checker
        self._status_callback = status_callback or (lambda _state: None)
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
            dtype=(
                torch_module.bfloat16 if config.device == "cuda:0" else torch_module.float32
            ),
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
            if self.config.device == "cuda:0" and not torch_module.cuda.is_available():
                raise RuntimeError("CUDA no está disponible")
            if self.config.device == "cuda:0" and not torch_module.cuda.is_bf16_supported():
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
        self._status_callback(state)

    def ensure_model(self) -> Any:
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
            self._set_state("model_loaded", None)
            return model
        except Exception as exc:
            self._set_state("error", str(exc))
            raise

    def _gpu_details(self) -> dict[str, object]:
        torch_module = self._get_torch()
        cuda = torch_module.cuda
        available = bool(self.config.device == "cuda:0" and cuda.is_available())
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
            model = self.ensure_model()
            if instruct and not self._supports_instruct:
                self._set_state("idle", None)
                raise QwenServiceError(
                    "unsupported_instruct",
                    "El modelo Qwen 0.6B instalado no admite instruct",
                )
            self._set_state("generating")
            torch_module = self._get_torch()
            seed = int(options.pop("seed"))
            devices = [0] if self.config.device == "cuda:0" else []
            with (
                torch_module.inference_mode(),
                torch_module.random.fork_rng(devices=devices, enabled=True),
            ):
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
            self._set_state("validating")
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
            before = self._gpu_details().get("vram_free_bytes")
            was_loaded = self._model is not None
            self._model = None
            gc.collect()
            if self._torch is not None and self.config.device == "cuda:0":
                self._torch.cuda.empty_cache()
                ipc_collect = getattr(self._torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    with contextlib.suppress(RuntimeError):
                        ipc_collect()
            after = self._gpu_details().get("vram_free_bytes")
            self._set_state("idle", None)
            return {
                "ok": True,
                "unloaded": was_loaded,
                "vram_free_before_bytes": before,
                "vram_free_after_bytes": after,
            }
        finally:
            self._generation_lock.release()


def _emit(stream: TextIO, payload: dict[str, object]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def run_worker(
    config: QwenServiceConfig,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> int:
    """Run commands until controlled shutdown or a fatal CUDA/model error."""

    active_request: str | None = None

    def status(state: str) -> None:
        _emit(
            output_stream,
            {"event": "status", "state": state, "request_id": active_request},
        )

    runtime = QwenRuntime(config, status_callback=status)
    try:
        runtime.initialize()
    except Exception as exc:
        _emit(output_stream, {"event": "fatal", "error": str(exc)})
        return 2
    _emit(output_stream, {"event": "ready", "pid": os.getpid(), "health": runtime.health()})
    for line in input_stream:
        try:
            command = json.loads(line)
            if not isinstance(command, dict):
                raise ValueError("El comando debe ser un objeto")
            active_request = str(command.get("request_id") or uuid4().hex)
            action = command.get("command")
            if action == "synthesize":
                result = runtime.synthesize(command.get("payload"))
            elif action == "unload":
                result = runtime.unload()
            elif action == "health":
                result = runtime.health()
            elif action == "capabilities":
                result = runtime.capabilities()
            elif action == "shutdown":
                result = runtime.unload()
                _emit(
                    output_stream,
                    {"event": "response", "request_id": active_request, "result": result},
                )
                return 0
            else:
                raise QwenServiceError("invalid_command", "Comando de worker desconocido")
            _emit(
                output_stream,
                {"event": "response", "request_id": active_request, "result": result},
            )
        except QwenServiceError as exc:
            _emit(
                output_stream,
                {
                    "event": "response",
                    "request_id": active_request,
                    "error": {
                        "code": exc.code,
                        "message": str(exc),
                        "retryable": exc.retryable,
                        "status": exc.status,
                    },
                },
            )
            if exc.code == "generation_failed":
                return 3
        except Exception as exc:
            _emit(output_stream, {"event": "fatal", "error": str(exc)})
            return 4
        finally:
            active_request = None
    with contextlib.suppress(Exception):
        runtime.unload()
    return 0


def main() -> None:
    raise SystemExit(run_worker(QwenServiceConfig.from_env()))


if __name__ == "__main__":
    main()
