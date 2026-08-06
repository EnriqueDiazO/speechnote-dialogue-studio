"""Fail-closed GPU safety models for the local Qwen controller.

The defaults are deliberately conservative for the audited RTX 3060 Ti workstation.
They are an editable local policy, not universal limits for other computers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GpuSafetyPolicy:
    """Validated thresholds and lifecycle limits applied by the controller."""

    enabled: bool = True
    fail_closed: bool = True
    require_cuda: bool = True
    require_bf16: bool = True
    max_gpu_util_percent: int = 75
    max_vram_used_percent: int = 70
    min_vram_free_mb: int = 3000
    max_temperature_c: int = 75
    recent_xid_window_seconds: int = 900
    block_on_recent_xid: bool = True
    block_on_display_engine_warning: bool = True
    block_when_worker_exists: bool = True
    worker_start_timeout_seconds: int = 30
    model_load_timeout_seconds: int = 180
    synthesis_timeout_seconds: int = 300
    idle_unload_seconds: int = 120
    idle_shutdown_seconds: int = 300
    post_job_cooldown_seconds: int = 5
    allow_cpu_fallback: bool = False
    monitor_interval_seconds: float = 1.5
    kernel_poll_interval_seconds: float = 10.0
    terminate_grace_seconds: float = 8.0

    def __post_init__(self) -> None:
        percent_fields = (
            "max_gpu_util_percent",
            "max_vram_used_percent",
        )
        for name in percent_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
                raise ValueError(f"{name} debe ser un entero entre 0 y 100")
        positive_int_fields = (
            "min_vram_free_mb",
            "max_temperature_c",
            "recent_xid_window_seconds",
            "worker_start_timeout_seconds",
            "model_load_timeout_seconds",
            "synthesis_timeout_seconds",
            "idle_unload_seconds",
            "idle_shutdown_seconds",
            "post_job_cooldown_seconds",
        )
        for name in positive_int_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} debe ser un entero no negativo")
        positive_number_fields = (
            "monitor_interval_seconds",
            "kernel_poll_interval_seconds",
            "terminate_grace_seconds",
        )
        for name in positive_number_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} debe ser mayor que cero")
        if self.idle_shutdown_seconds and (
            self.idle_unload_seconds > self.idle_shutdown_seconds
        ):
            raise ValueError("idle_unload_seconds no puede exceder idle_shutdown_seconds")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> GpuSafetyPolicy:
        known = {item.name for item in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Opciones de seguridad desconocidas: {', '.join(sorted(unknown))}")
        return cls(**dict(values))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    process_name: str
    used_memory_mb: int | None = None
    process_type: str | None = None
    recognized: bool = False


@dataclass(frozen=True)
class GpuPreflightResult:
    """Privacy-conscious snapshot used to allow or block Qwen."""

    allowed: bool
    timestamp: str
    gpu_name: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    temperature_c: int | None = None
    gpu_util_percent: int | None = None
    vram_total_mb: int | None = None
    vram_used_mb: int | None = None
    vram_free_mb: int | None = None
    compute_processes: tuple[GpuProcess, ...] = ()
    graphics_processes: tuple[GpuProcess, ...] = ()
    recent_kernel_events: tuple[str, ...] = ()
    recent_xid_events: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    recommended_actions: tuple[str, ...] = ()
    session_type: str | None = None
    display: str | None = None
    cuda_available: bool | None = None
    bf16_available: bool | None = None
    runtime_torch_version: str | None = None
    data_sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def blocked(cls, blocker: str, **values: Any) -> GpuPreflightResult:
        return cls(
            allowed=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            blockers=(blocker,),
            **values,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["compute_processes"] = [asdict(item) for item in self.compute_processes]
        data["graphics_processes"] = [asdict(item) for item in self.graphics_processes]
        return data


@dataclass(frozen=True)
class GpuMetricSnapshot:
    timestamp: str
    phase: str
    temperature_c: int | None
    gpu_util_percent: int | None
    vram_used_mb: int | None
    vram_free_mb: int | None
    worker_alive: bool
    new_xid_events: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_gpu_safety_policy(path: Path | None = None) -> GpuSafetyPolicy:
    """Load an optional JSON policy; missing files keep audited defaults."""

    if path is None or not path.exists():
        return GpuSafetyPolicy()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"No se pudo leer la política GPU: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("La política GPU debe ser un objeto JSON")
    return GpuSafetyPolicy.from_mapping(payload)


def policy_json(policy: GpuSafetyPolicy) -> str:
    return json.dumps(policy.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
