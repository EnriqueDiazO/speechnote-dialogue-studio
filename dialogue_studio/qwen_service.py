"""Local Qwen controller that never imports Torch or ``qwen_tts``.

The HTTP controller owns policy, queue and lifecycle. CUDA exists only in the child
``dialogue_studio.qwen_worker`` process created with subprocess spawn semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

from . import __version__
from .qwen_gpu_safety import GpuMetricSnapshot, GpuPreflightResult, GpuSafetyPolicy
from .qwen_preflight import collect_gpu_metrics, collect_kernel_events, run_gpu_preflight

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
    """Structured controller error safe to return to a localhost client."""

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
    policy: GpuSafetyPolicy = field(default_factory=GpuSafetyPolicy)

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
        cuda_mode = device == "cuda:0" and dtype == "bfloat16"
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
        policy_payload = values.get("QWEN_GPU_SAFETY_POLICY")
        policy = GpuSafetyPolicy()
        if policy_payload:
            decoded = json.loads(policy_payload)
            if not isinstance(decoded, dict):
                raise ValueError("QWEN_GPU_SAFETY_POLICY debe ser un objeto JSON")
            policy = GpuSafetyPolicy.from_mapping(decoded)
        cpu_mode = (
            device == "cpu"
            and dtype == "float32"
            and values.get("QWEN_CPU_EMERGENCY_CONFIRMED") == "1"
            and policy.allow_cpu_fallback
        )
        if attention != "sdpa" or not (cuda_mode or cpu_mode):
            raise ValueError(
                "Qwen requiere cuda:0/bfloat16 o CPU float32 expresamente autorizado, con sdpa"
            )
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
            policy=policy,
        )

    @property
    def worker_log_file(self) -> Path:
        return self.pid_file.parent / "qwen-worker.log"

    @property
    def controller_log_file(self) -> Path:
        return self.pid_file.parent / "qwen-controller.jsonl"


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


@dataclass
class QwenJob:
    request_id: str
    payload: dict[str, object]
    stage: str = "pending"
    result: dict[str, object] | None = None
    error: QwenServiceError | None = None
    canceled: bool = False
    done: threading.Event = field(default_factory=threading.Event)


@dataclass
class WorkerHandle:
    process: Any
    messages: queue.Queue[dict[str, Any]]
    reader: threading.Thread


class QwenController:
    """Serialize jobs and own exactly one disposable CUDA worker."""

    creation_method = "subprocess_spawn"

    def __init__(
        self,
        config: QwenServiceConfig,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        preflight_runner: Callable[..., GpuPreflightResult] = run_gpu_preflight,
        metric_sampler: Callable[[], dict[str, object]] = collect_gpu_metrics,
        kernel_sampler: Callable[[int], tuple[str, ...]] = collect_kernel_events,
    ) -> None:
        self.config = config
        self.policy = config.policy
        self._popen = popen
        self._preflight_runner = preflight_runner
        self._metric_sampler = metric_sampler
        self._kernel_sampler = kernel_sampler
        self._lock = threading.RLock()
        self._worker_io_lock = threading.Lock()
        self._jobs: queue.Queue[QwenJob | None] = queue.Queue()
        self._known_jobs: dict[str, QwenJob] = {}
        self._worker: WorkerHandle | None = None
        self._current_job: QwenJob | None = None
        self._state = "starting"
        self._last_error: str | None = None
        self._last_preflight: GpuPreflightResult | None = None
        self._last_worker_exit_code: int | None = None
        self._gpu_fault_reason: str | None = None
        self._gpu_metrics: list[GpuMetricSnapshot] = []
        self._baseline_kernel_events: set[str] = set()
        self._last_activity_monotonic = time.monotonic()
        self._last_job_timestamp: str | None = None
        self._last_unload_result: dict[str, object] | None = None
        self._last_timeout: dict[str, object] | None = None
        self._errors: list[dict[str, object]] = []
        self._model_loaded = False
        self._worker_mode: str | None = None
        self._load_count = 0
        self._capabilities = self._fallback_capabilities()
        self._closing = threading.Event()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="qwen-job-dispatcher",
            daemon=True,
        )
        self._lifecycle = threading.Thread(
            target=self._lifecycle_loop,
            name="qwen-worker-lifecycle",
            daemon=True,
        )

    def start(self) -> None:
        if not self._dispatcher.is_alive():
            self._state = "idle"
            self._dispatcher.start()
            self._lifecycle.start()
            self._log("controller_started", creation_method=self.creation_method)

    def _log(self, event: str, **details: object) -> None:
        record = {
            "timestamp": time.time(),
            "event": event,
            **details,
        }
        try:
            self.config.controller_log_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with self.config.controller_log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError:
            return

    @staticmethod
    def _fallback_capabilities() -> dict[str, object]:
        return {
            "ok": True,
            "speakers": list(FALLBACK_SPEAKERS),
            "languages": list(FALLBACK_LANGUAGES),
            "source": "verified_fallback",
            "supports_instruct": False,
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

    def _worker_pid(self) -> int | None:
        worker = self._worker
        if worker is None or worker.process.poll() is not None:
            return None
        return int(worker.process.pid)

    def preflight(self, *, queued_job: bool = False) -> GpuPreflightResult:
        result = self._preflight_runner(
            self.policy,
            Path(sys.executable),
            recognized_worker_pid=self._worker_pid(),
            synthesis_in_progress=self._current_job is not None and not queued_job,
            service_state="idle",
        )
        with self._lock:
            self._last_preflight = result
            self._baseline_kernel_events = set(result.recent_kernel_events)
            if not result.allowed and self._state not in {"gpu_fault", "stopping"}:
                self._state = "blocked"
                self._last_error = result.blockers[0] if result.blockers else "Preflight bloqueado"
            elif result.allowed and self._state == "blocked":
                self._state = "idle"
                self._last_error = None
        self._log("preflight", allowed=result.allowed, blockers=list(result.blockers))
        return result

    def health(self) -> dict[str, object]:
        worker_pid = self._worker_pid()
        idle_seconds = max(0.0, time.monotonic() - self._last_activity_monotonic)
        with self._lock:
            queue_items = [
                {"request_id": job.request_id, "stage": job.stage}
                for job in self._known_jobs.values()
                if not job.done.is_set()
            ]
            return {
                "ok": self._state not in {"error", "gpu_fault"},
                "service": "speechnote-dialogue-studio-qwen",
                "state": self._state,
                "model": self.config.model,
                "model_loaded": self._model_loaded,
                "load_count": self._load_count,
                "device": self.config.device,
                "dtype": self.config.dtype,
                "attention": self.config.attention,
                "native_sample_rate": 24_000,
                "last_error": self._last_error,
                "worker_pid": worker_pid,
                "worker_alive": worker_pid is not None,
                "worker_creation_method": self.creation_method,
                "worker_mode": self._worker_mode,
                "worker_exit_code": self._last_worker_exit_code,
                "queue": queue_items,
                "current_stage": self._current_job.stage if self._current_job else None,
                "policy": self.policy.to_dict(),
                "preflight": self._last_preflight.to_dict() if self._last_preflight else None,
                "gpu_fault_reason": self._gpu_fault_reason,
                "latest_gpu_metric": (
                    self._gpu_metrics[-1].to_dict() if self._gpu_metrics else None
                ),
                "gpu_metric_count": len(self._gpu_metrics),
                "last_job_timestamp": self._last_job_timestamp,
                "idle_seconds": idle_seconds,
                "last_unload": self._last_unload_result,
            }

    def capabilities(self) -> dict[str, object]:
        return dict(self._capabilities)

    @staticmethod
    def _sha256(path: Path) -> str | None:
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except OSError:
            return None

    def _diagnostic_outputs(self) -> list[dict[str, object]]:
        outputs: list[dict[str, object]] = []
        root = self.config.output_root.resolve()
        for job in self._known_jobs.values():
            raw = job.payload.get("output_path")
            if not isinstance(raw, str):
                continue
            try:
                path = Path(raw).resolve(strict=False)
                relative = path.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            outputs.append(
                {
                    "request_id": job.request_id,
                    "stage": job.stage,
                    "relative_path": relative,
                    "sha256": self._sha256(path),
                }
            )
        return outputs

    def diagnostic(self) -> dict[str, object]:
        """Return a privacy-safe diagnostic; never include script text or audio bytes."""

        preflight = self._last_preflight.to_dict() if self._last_preflight else None
        return {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "application_version": __version__,
            "python_version": sys.version.split()[0],
            "runtime": {
                "torch": preflight.get("runtime_torch_version") if preflight else None,
                "cuda": preflight.get("cuda_version") if preflight else None,
                "driver": preflight.get("driver_version") if preflight else None,
                "gpu": preflight.get("gpu_name") if preflight else None,
                "dtype": self.config.dtype,
                "attention": self.config.attention,
            },
            "policy": self.policy.to_dict(),
            "preflight": preflight,
            "metrics": [item.to_dict() for item in self._gpu_metrics],
            "worker": {
                "pid": self._worker_pid(),
                "alive": self._worker_pid() is not None,
                "creation_method": self.creation_method,
                "mode": self._worker_mode,
                "model_loaded": self._model_loaded,
                "load_count": self._load_count,
                "exit_code": self._last_worker_exit_code,
                "state": self._state,
            },
            "timeout": self._last_timeout,
            "errors": list(self._errors),
            "gpu_fault": self._gpu_fault_reason,
            "outputs": self._diagnostic_outputs(),
        }

    def _reader_loop(
        self,
        process: Any,
        messages: queue.Queue[dict[str, Any]],
    ) -> None:
        if process.stdout is None:
            messages.put({"event": "process_exit", "exit_code": process.poll()})
            return
        for line in process.stdout:
            try:
                message = json.loads(line)
                if isinstance(message, dict):
                    messages.put(message)
                    continue
            except json.JSONDecodeError:
                pass
            self._log("worker_output", line=line.rstrip()[:1000])
        messages.put({"event": "process_exit", "exit_code": process.poll()})

    def _start_worker(self, mode: str = "cuda") -> None:
        if mode not in {"cuda", "cpu"}:
            raise QwenServiceError("invalid_mode", "Modo de ejecución Qwen desconocido")
        if self._worker_pid() is not None and self._worker_mode == mode:
            return
        if self._worker_pid() is not None:
            self._terminate_worker("execution_mode_change")
        self._state = "starting_worker"
        environment = os.environ.copy()
        if mode == "cpu":
            environment.update(
                {
                    "QWEN_TTS_DEVICE": "cpu",
                    "QWEN_TTS_DTYPE": "float32",
                    "QWEN_CPU_EMERGENCY_CONFIRMED": "1",
                    "QWEN_GPU_SAFETY_POLICY": json.dumps(
                        self.policy.to_dict(), separators=(",", ":")
                    ),
                }
            )
        else:
            environment.update(
                {
                    "QWEN_TTS_DEVICE": "cuda:0",
                    "QWEN_TTS_DTYPE": "bfloat16",
                    "QWEN_CPU_EMERGENCY_CONFIRMED": "0",
                }
            )
        self.config.worker_log_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self.config.worker_log_file.open("ab", buffering=0) as error_log:
            process = self._popen(
                [sys.executable, "-m", "dialogue_studio.qwen_worker"],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=error_log,
                text=True,
                bufsize=1,
                start_new_session=True,
                close_fds=True,
            )
        messages: queue.Queue[dict[str, Any]] = queue.Queue()
        reader = threading.Thread(
            target=self._reader_loop,
            args=(process, messages),
            name=f"qwen-worker-reader-{process.pid}",
            daemon=True,
        )
        self._worker = WorkerHandle(process, messages, reader)
        self._worker_mode = mode
        reader.start()
        deadline = time.monotonic() + self.policy.worker_start_timeout_seconds
        while time.monotonic() < deadline:
            remaining = max(0.01, deadline - time.monotonic())
            try:
                message = messages.get(timeout=min(0.2, remaining))
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if message.get("event") == "ready":
                health = message.get("health")
                if isinstance(health, dict):
                    self._model_loaded = bool(health.get("model_loaded"))
                self._state = "idle"
                self._last_error = None
                self._log("worker_started", worker_pid=process.pid)
                return
            if message.get("event") in {"fatal", "process_exit"}:
                self._last_error = str(message.get("error") or "El worker terminó al iniciar")
                break
        self._terminate_worker("worker_start_timeout")
        raise QwenServiceError(
            "worker_start_timeout",
            "El worker Qwen no inició dentro del tiempo permitido",
            status=HTTPStatus.GATEWAY_TIMEOUT,
            retryable=True,
        )

    def _write_worker(self, message: dict[str, object]) -> None:
        worker = self._worker
        if worker is None or worker.process.poll() is not None or worker.process.stdin is None:
            raise QwenServiceError(
                "worker_dead",
                "El worker Qwen no está disponible",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                retryable=True,
            )
        try:
            worker.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            worker.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise QwenServiceError(
                "worker_dead",
                "El worker Qwen terminó inesperadamente",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                retryable=True,
            ) from exc

    def _handle_status(self, message: dict[str, Any], job: QwenJob | None = None) -> str:
        state = str(message.get("state") or "")
        stages = {
            "loading_model": "loading_model",
            "model_loaded": "loading_model",
            "generating": "generating",
            "validating": "validating",
            "idle": "finalized",
        }
        if job is not None and state in stages:
            job.stage = stages[state]
        if state == "model_loaded":
            self._model_loaded = True
            self._load_count += 1
        if state:
            self._state = state
        return state

    def _run_synthesis(self, job: QwenJob, *, mode: str = "cuda") -> dict[str, object]:
        worker = self._worker
        if worker is None:
            raise QwenServiceError("worker_dead", "No existe worker Qwen")
        request_id = job.request_id
        job.stage = "loading_model" if not self._model_loaded else "generating"
        phase = job.stage
        timeout = (
            self.policy.model_load_timeout_seconds
            if phase == "loading_model"
            else self.policy.synthesis_timeout_seconds
        )
        deadline = time.monotonic() + timeout
        next_metric_poll = time.monotonic()
        next_kernel_poll = time.monotonic() + self.policy.kernel_poll_interval_seconds
        self._write_worker(
            {"request_id": request_id, "command": "synthesize", "payload": job.payload}
        )
        while True:
            if job.canceled:
                self._terminate_worker("job_canceled")
                raise QwenServiceError("canceled", "La síntesis Qwen fue cancelada")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                code = "model_load_timeout" if phase == "loading_model" else "synthesis_timeout"
                self._last_timeout = {
                    "code": code,
                    "phase": phase,
                    "limit_seconds": (
                        self.policy.model_load_timeout_seconds
                        if phase == "loading_model"
                        else self.policy.synthesis_timeout_seconds
                    ),
                }
                self._terminate_worker(code)
                raise QwenServiceError(
                    code,
                    "Qwen excedió el tiempo permitido durante "
                    + ("la carga del modelo" if phase == "loading_model" else "la síntesis"),
                    status=HTTPStatus.GATEWAY_TIMEOUT,
                    retryable=True,
                )
            try:
                now = time.monotonic()
                until_monitor = max(0.01, next_metric_poll - now)
                message = worker.messages.get(timeout=min(until_monitor, remaining))
            except queue.Empty:
                if worker.process.poll() is not None:
                    self._worker_exited(worker)
                    raise QwenServiceError(
                        "worker_dead",
                        "El worker Qwen terminó durante la síntesis",
                        status=HTTPStatus.SERVICE_UNAVAILABLE,
                        retryable=True,
                    ) from None
                message = {}
            now = time.monotonic()
            if mode == "cuda" and now >= next_metric_poll:
                poll_kernel = now >= next_kernel_poll
                blocker = self._monitor_worker(job.stage, poll_kernel=poll_kernel)
                next_metric_poll = now + self.policy.monitor_interval_seconds
                if poll_kernel:
                    next_kernel_poll = now + self.policy.kernel_poll_interval_seconds
                if blocker:
                    self._set_gpu_fault(blocker)
                    raise QwenServiceError(
                        "gpu_fault",
                        blocker,
                        status=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
            if not message:
                continue
            event = message.get("event")
            if event == "status" and message.get("request_id") in {request_id, None}:
                state = self._handle_status(message, job)
                if state == "generating" and phase != "generating":
                    phase = "generating"
                    deadline = time.monotonic() + self.policy.synthesis_timeout_seconds
                continue
            if event == "response" and message.get("request_id") == request_id:
                error = message.get("error")
                if isinstance(error, dict):
                    raise QwenServiceError(
                        str(error.get("code") or "worker_error"),
                        str(error.get("message") or "El worker Qwen falló"),
                        status=int(error.get("status") or HTTPStatus.INTERNAL_SERVER_ERROR),
                        retryable=bool(error.get("retryable")),
                    )
                result = message.get("result")
                if not isinstance(result, dict):
                    raise QwenServiceError("worker_protocol", "Respuesta inválida del worker")
                self._state = "idle"
                return result
            if event in {"fatal", "process_exit"}:
                self._worker_exited(worker)
                raise QwenServiceError(
                    "worker_dead",
                    str(message.get("error") or "El worker Qwen terminó inesperadamente"),
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                    retryable=True,
                )

    def _monitor_worker(self, phase: str, *, poll_kernel: bool) -> str | None:
        worker_alive = self._worker_pid() is not None
        try:
            metrics = self._metric_sampler()
        except RuntimeError as exc:
            snapshot = GpuMetricSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                phase=phase,
                temperature_c=None,
                gpu_util_percent=None,
                vram_used_mb=None,
                vram_free_mb=None,
                worker_alive=worker_alive,
            )
            self._gpu_metrics.append(snapshot)
            self._gpu_metrics = self._gpu_metrics[-1000:]
            return str(exc) if self.policy.fail_closed else None

        new_xids: tuple[str, ...] = ()
        if poll_kernel:
            try:
                events = self._kernel_sampler(self.policy.recent_xid_window_seconds)
            except RuntimeError as exc:
                if self.policy.fail_closed:
                    return str(exc)
            else:
                new_xids = tuple(
                    event
                    for event in events
                    if "xid" in event.lower() and event not in self._baseline_kernel_events
                )
                self._baseline_kernel_events.update(events)
        temperature = metrics.get("temperature_c")
        utilization = metrics.get("gpu_util_percent")
        total = metrics.get("vram_total_mb")
        used = metrics.get("vram_used_mb")
        free = metrics.get("vram_free_mb")
        snapshot = GpuMetricSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            phase=phase,
            temperature_c=temperature if isinstance(temperature, int) else None,
            gpu_util_percent=utilization if isinstance(utilization, int) else None,
            vram_used_mb=used if isinstance(used, int) else None,
            vram_free_mb=free if isinstance(free, int) else None,
            worker_alive=worker_alive,
            new_xid_events=new_xids,
        )
        self._gpu_metrics.append(snapshot)
        self._gpu_metrics = self._gpu_metrics[-1000:]
        if new_xids:
            return (
                "Se detectó un Xid nuevo durante la inferencia. No se iniciarán más trabajos; "
                "reinicia la sesión gráfica o el sistema según su estado y exporta el diagnóstico."
            )
        if isinstance(temperature, int) and temperature > self.policy.max_temperature_c:
            return (
                f"Temperatura GPU crítica durante inferencia: {temperature} °C "
                f"> {self.policy.max_temperature_c} °C"
            )
        critical_free_mb = max(256, self.policy.min_vram_free_mb // 4)
        if isinstance(free, int) and free < critical_free_mb:
            return (
                f"VRAM libre crítica durante inferencia: {free} MiB "
                f"< {critical_free_mb} MiB"
            )
        if (
            isinstance(used, int)
            and isinstance(total, int)
            and total > 0
            and (used / total) * 100 >= 95
        ):
            return "VRAM usada alcanzó el umbral crítico de 95% durante inferencia"
        return None

    def _set_gpu_fault(self, reason: str) -> None:
        self._gpu_fault_reason = reason
        self._last_error = reason
        self._state = "gpu_fault"
        self._terminate_worker("gpu_fault")
        self._state = "gpu_fault"
        self._log("gpu_fault", reason=reason)

    def _worker_exited(self, worker: WorkerHandle) -> None:
        self._last_worker_exit_code = worker.process.poll()
        if self._worker is worker:
            self._worker = None
        self._model_loaded = False
        self._worker_mode = None

    def _command_worker(self, action: str, *, timeout: float = 30) -> dict[str, object]:
        worker = self._worker
        if worker is None or self._worker_pid() is None:
            return {"ok": True, "worker_running": False}
        request_id = uuid4().hex
        with self._worker_io_lock:
            self._write_worker({"request_id": request_id, "command": action})
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    message = worker.messages.get(timeout=0.2)
                except queue.Empty:
                    if worker.process.poll() is not None:
                        self._worker_exited(worker)
                        break
                    continue
                if message.get("event") == "status":
                    self._handle_status(message)
                    continue
                if (
                    message.get("event") == "response"
                    and message.get("request_id") == request_id
                ):
                    result = message.get("result")
                    if isinstance(result, dict):
                        if action == "unload":
                            self._model_loaded = False
                            self._last_unload_result = result
                        return result
                if message.get("event") in {"fatal", "process_exit"}:
                    self._worker_exited(worker)
                    break
        raise QwenServiceError(
            "worker_timeout",
            f"El worker no respondió a {action}",
            status=HTTPStatus.GATEWAY_TIMEOUT,
            retryable=True,
        )

    def _terminate_worker(self, reason: str) -> dict[str, object]:
        worker = self._worker
        if worker is None:
            self._model_loaded = False
            self._worker_mode = None
            return {"ok": True, "stopped": False, "reason": reason}
        process = worker.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.policy.terminate_grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._last_worker_exit_code = process.poll()
        self._worker = None
        self._model_loaded = False
        self._worker_mode = None
        self._log(
            "worker_stopped",
            reason=reason,
            worker_pid=getattr(process, "pid", None),
            exit_code=self._last_worker_exit_code,
        )
        if self._state not in {"gpu_fault", "stopping"}:
            self._state = "idle"
        return {"ok": True, "stopped": True, "reason": reason}

    def unload(self) -> dict[str, object]:
        if self._current_job is not None:
            raise QwenServiceError(
                "gpu_busy",
                "No se puede descargar el modelo durante una síntesis",
                status=HTTPStatus.CONFLICT,
                retryable=True,
            )
        result = self._command_worker("unload")
        self._last_unload_result = result
        return result

    def stop_worker(self) -> dict[str, object]:
        if self._current_job is not None:
            raise QwenServiceError(
                "gpu_busy",
                "Cancela la síntesis actual antes de detener el worker",
                status=HTTPStatus.CONFLICT,
                retryable=True,
            )
        return self._terminate_worker("manual_stop")

    def submit(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise QwenServiceError("invalid_request", "El cuerpo debe ser un objeto JSON")
        if self._closing.is_set():
            raise QwenServiceError(
                "service_stopping",
                "El servicio Qwen se está deteniendo",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        job = QwenJob(uuid4().hex, payload)
        with self._lock:
            self._known_jobs[job.request_id] = job
        self._jobs.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        if job.result is None:
            raise QwenServiceError("worker_protocol", "El trabajo Qwen no devolvió resultado")
        return job.result

    def cancel(self, *, stop_current: bool = True) -> dict[str, object]:
        canceled = 0
        with self._lock:
            for job in self._known_jobs.values():
                if job is self._current_job:
                    if stop_current:
                        job.canceled = True
                    continue
                if not job.done.is_set():
                    job.canceled = True
                    job.stage = "canceled"
                    job.error = QwenServiceError("canceled", "Trabajo Qwen cancelado")
                    job.done.set()
                    canceled += 1
        return {
            "ok": True,
            "canceled_pending": canceled,
            "current_cancel_requested": bool(stop_current and self._current_job),
        }

    def _dispatch_loop(self) -> None:
        while not self._closing.is_set():
            job = self._jobs.get()
            if job is None:
                return
            if job.canceled or job.done.is_set():
                continue
            with self._lock:
                self._current_job = job
            try:
                mode = str(job.payload.get("execution_mode") or "cuda")
                cpu_confirmed = job.payload.get("confirm_cpu_fallback") is True
                if mode not in {"cuda", "cpu"}:
                    raise QwenServiceError("invalid_mode", "Modo de ejecución Qwen desconocido")
                if mode == "cpu" and not (
                    self.policy.allow_cpu_fallback and cpu_confirmed
                ):
                    raise QwenServiceError(
                        "cpu_fallback_not_confirmed",
                        "El modo CPU requiere habilitación en la política y confirmación explícita",
                    )
                if mode == "cuda" and self._gpu_fault_reason:
                    raise QwenServiceError(
                        "gpu_fault",
                        "Qwen permanece bloqueado después de un fallo GPU: "
                        + self._gpu_fault_reason,
                        status=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                job.stage = "preflight"
                if mode == "cuda":
                    preflight = self.preflight(queued_job=True)
                    if not preflight.allowed:
                        detail = (
                            preflight.blockers[0]
                            if preflight.blockers
                            else "estado no seguro"
                        )
                        raise QwenServiceError(
                            "gpu_preflight_blocked",
                            "Generación bloqueada para proteger la sesión gráfica. " + detail,
                            status=HTTPStatus.SERVICE_UNAVAILABLE,
                        )
                else:
                    self._state = "cpu_emergency"
                job.stage = "starting_worker"
                self._start_worker(mode)
                with self._worker_io_lock:
                    job.result = self._run_synthesis(job, mode=mode)
                    job.result["execution_mode"] = mode
                job.stage = "finalized"
                if self.policy.post_job_cooldown_seconds:
                    time.sleep(self.policy.post_job_cooldown_seconds)
            except QwenServiceError as exc:
                job.stage = "canceled" if exc.code == "canceled" else "error"
                job.error = exc
                self._last_error = str(exc)
                self._errors.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "code": exc.code,
                        "message": str(exc).replace(str(Path.home()), "[HOME]"),
                    }
                )
                self._errors = self._errors[-100:]
                if self._state not in {"gpu_fault", "blocked"}:
                    self._state = "idle" if exc.retryable else "error"
            except Exception as exc:
                job.stage = "error"
                job.error = QwenServiceError(
                    "controller_error", str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self._last_error = str(exc)
                self._state = "error"
            finally:
                with self._lock:
                    self._current_job = None
                    self._last_activity_monotonic = time.monotonic()
                    self._last_job_timestamp = datetime.now(timezone.utc).isoformat()
                job.done.set()

    def _lifecycle_tick(self, now: float | None = None) -> str | None:
        current = time.monotonic() if now is None else now
        if self._current_job is not None or self._worker_pid() is None:
            return None
        idle = max(0.0, current - self._last_activity_monotonic)
        if (
            self._model_loaded
            and self.policy.idle_unload_seconds > 0
            and idle >= self.policy.idle_unload_seconds
        ):
            try:
                result = self._command_worker("unload")
            except QwenServiceError as exc:
                self._log("idle_unload_failed", error=str(exc))
                return "idle_unload_failed"
            self._last_unload_result = result
            self._log("idle_unload", **result)
            return "idle_unload"
        if (
            self.policy.idle_shutdown_seconds > 0
            and idle >= self.policy.idle_shutdown_seconds
        ):
            self._terminate_worker("idle_shutdown")
            return "idle_shutdown"
        return None

    def _lifecycle_loop(self) -> None:
        while not self._closing.wait(timeout=0.5):
            self._lifecycle_tick()

    def close(self) -> None:
        self._state = "stopping"
        self._closing.set()
        self.cancel(stop_current=True)
        self._terminate_worker("controller_shutdown")
        self._jobs.put(None)
        if self._dispatcher.is_alive() and threading.current_thread() is not self._dispatcher:
            self._dispatcher.join(timeout=2)
        if self._lifecycle.is_alive() and threading.current_thread() is not self._lifecycle:
            self._lifecycle.join(timeout=2)
        self._state = "offline"
        self._log("controller_stopped")


class QwenHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], controller: QwenController):
        super().__init__(address, QwenRequestHandler)
        self.controller = controller


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
                "state": self.server.controller.health()["state"],
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
                self._send(HTTPStatus.OK, self.server.controller.health())
            elif self.path == "/capabilities":
                self._send(HTTPStatus.OK, self.server.controller.capabilities())
            elif self.path == "/preflight":
                self._send(HTTPStatus.OK, self.server.controller.preflight().to_dict())
            elif self.path == "/diagnostic":
                self._send(HTTPStatus.OK, self.server.controller.diagnostic())
            else:
                raise QwenServiceError("not_found", "Endpoint desconocido", status=404)
        except QwenServiceError as exc:
            self._error(exc)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/synthesize":
                self._send(HTTPStatus.OK, self.server.controller.submit(self._payload()))
            elif self.path == "/unload":
                self._send(HTTPStatus.OK, self.server.controller.unload())
            elif self.path == "/worker/stop":
                self._send(HTTPStatus.OK, self.server.controller.stop_worker())
            elif self.path == "/cancel":
                payload = self._payload()
                stop_current = (
                    not isinstance(payload, dict) or payload.get("stop_current") is not False
                )
                self._send(
                    HTTPStatus.OK,
                    self.server.controller.cancel(stop_current=stop_current),
                )
            elif self.path == "/shutdown":
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
    controller = QwenController(config)
    try:
        controller.start()
        server = QwenHTTPServer((config.host, config.port), controller)

        def stop(_signum: int, _frame: object) -> None:
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        server.serve_forever(poll_interval=0.25)
    finally:
        controller.close()
        if server is not None:
            server.server_close()
        _release_pid_file(config.pid_file)


def main() -> None:
    argparse.ArgumentParser(description="Controller local aislado Qwen3-TTS").parse_args()
    serve(QwenServiceConfig.from_env())


if __name__ == "__main__":
    main()
