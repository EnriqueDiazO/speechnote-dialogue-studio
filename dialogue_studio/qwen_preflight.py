"""Unprivileged NVIDIA, CUDA and graphical-session preflight for Qwen."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .qwen_gpu_safety import GpuPreflightResult, GpuProcess, GpuSafetyPolicy

GPU_QUERY = (
    "name,driver_version,temperature.gpu,utilization.gpu,"
    "memory.total,memory.used,memory.free"
)
KERNEL_TERMS = (
    "nvrm",
    "xid",
    "nvidia-modeset",
    "fallen off",
    "mmu fault",
    "gpu has fallen",
    "failed to idle",
    "display engine",
    "hang",
    "timeout",
)
DISPLAY_RISK_TERMS = (
    "display engine",
    "failed to idle",
    "nvidia-modeset",
    "fallen off",
    "gpu has fallen",
    "mmu fault",
    "hang",
    "timeout",
)
GRAPHICAL_APP_TERMS = ("code", "electron", "firefox", "speech note", "speech-note")

CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def _run_command(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _clean_process_name(value: str) -> str:
    name = value.strip()
    try:
        path = Path(name)
        return path.name or name
    except (OSError, ValueError):
        return name


def _parse_gpu_row(output: str) -> dict[str, object]:
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError("Se requiere exactamente una GPU NVIDIA")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 7:
        raise ValueError("nvidia-smi devolvió métricas incompletas")
    return {
        "gpu_name": parts[0],
        "driver_version": parts[1],
        "temperature_c": int(parts[2]),
        "gpu_util_percent": int(parts[3]),
        "vram_total_mb": int(parts[4]),
        "vram_used_mb": int(parts[5]),
        "vram_free_mb": int(parts[6]),
    }


def _parse_compute_processes(output: str) -> list[GpuProcess]:
    processes: list[GpuProcess] = []
    for row in output.splitlines():
        if not row.strip() or "No running processes" in row:
            continue
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3:
            continue
        try:
            processes.append(
                GpuProcess(
                    pid=int(parts[0]),
                    process_name=_clean_process_name(parts[1]),
                    used_memory_mb=int(parts[2]),
                    process_type="compute",
                )
            )
        except ValueError:
            continue
    return processes


def _parse_graphics_processes(output: str) -> list[GpuProcess]:
    processes: list[GpuProcess] = []
    pattern = re.compile(r"\|\s*\d+\s+\S+\s+\S+\s+(\d+)\s+([CG+]+)\s+(.+?)\s+(\d+)MiB\s*\|")
    for match in pattern.finditer(output):
        process_type = match.group(2)
        if "G" not in process_type:
            continue
        processes.append(
            GpuProcess(
                pid=int(match.group(1)),
                process_name=_clean_process_name(match.group(3)),
                used_memory_mb=int(match.group(4)),
                process_type="graphics",
            )
        )
    return processes


def _filtered_kernel_events(output: str) -> list[str]:
    events: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        lowered = line.lower()
        if line and any(term in lowered for term in KERNEL_TERMS):
            events.append(line[:1000])
    return events[-100:]


def _runtime_probe_command(runtime_python: Path) -> list[str]:
    script = (
        "import json, torch; "
        "print(json.dumps({'torch': str(torch.__version__), "
        "'cuda_version': str(torch.version.cuda or ''), "
        "'cuda_available': bool(torch.cuda.is_available()), "
        "'bf16_available': bool(torch.cuda.is_available() and "
        "torch.cuda.is_bf16_supported())}))"
    )
    return [str(runtime_python), "-c", script]


def _scan_gpu_app_processes(proc_root: Path = Path("/proc")) -> tuple[list[str], list[int]]:
    accelerated: set[str] = set()
    qwen_workers: list[int] = []
    try:
        candidates = list(proc_root.iterdir())
    except OSError:
        return [], []
    for entry in candidates:
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", "replace"
            )
        except OSError:
            continue
        lowered = command.lower()
        if "dialogue_studio.qwen_worker" in lowered:
            qwen_workers.append(int(entry.name))
        if (
            "--type=gpu-process" in lowered
            or any(term in lowered for term in GRAPHICAL_APP_TERMS)
        ) and "--disable-gpu" not in lowered:
            label = next(
                (term for term in GRAPHICAL_APP_TERMS if term in lowered), "gpu-process"
            )
            accelerated.add(label)
    return sorted(accelerated), sorted(qwen_workers)


def _source_failure(
    source: str,
    result: subprocess.CompletedProcess[str] | None,
    error: Exception | None,
) -> str:
    if error is not None:
        return f"{source}: {type(error).__name__}"
    if result is None:
        return f"{source}: sin respuesta"
    detail = (result.stderr or result.stdout).strip().splitlines()
    suffix = detail[-1][:200] if detail else f"código {result.returncode}"
    return f"{source}: {suffix}"


def run_gpu_preflight(
    policy: GpuSafetyPolicy,
    runtime_python: Path,
    *,
    recognized_worker_pid: int | None = None,
    synthesis_in_progress: bool = False,
    service_state: str = "idle",
    environ: Mapping[str, str] | None = None,
    runner: CommandRunner = _run_command,
    proc_root: Path = Path("/proc"),
) -> GpuPreflightResult:
    """Collect current safety evidence and fail closed when it is incomplete."""

    timestamp = datetime.now(timezone.utc).isoformat()
    env = os.environ if environ is None else environ
    blockers: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []
    sources: dict[str, str] = {}
    metrics: dict[str, object] = {}
    compute_processes: list[GpuProcess] = []
    graphics_processes: list[GpuProcess] = []
    kernel_events: list[str] = []
    runtime: dict[str, Any] = {}

    def command(source: str, argv: Sequence[str], timeout: float = 10) -> str | None:
        result: subprocess.CompletedProcess[str] | None = None
        error: Exception | None = None
        try:
            result = runner(argv, timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            error = exc
        if error is not None or result is None or result.returncode != 0:
            sources[source] = _source_failure(source, result, error)
            return None
        sources[source] = "ok"
        return result.stdout

    gpu_output = command(
        "nvidia_smi_gpu",
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ],
    )
    if gpu_output is not None:
        try:
            metrics = _parse_gpu_row(gpu_output)
        except (TypeError, ValueError) as exc:
            sources["nvidia_smi_gpu"] = f"datos inválidos: {exc}"

    compute_output = command(
        "nvidia_smi_compute",
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    if compute_output is not None:
        compute_processes = _parse_compute_processes(compute_output)

    full_output = command("nvidia_smi_processes", ["nvidia-smi"])
    if full_output is not None:
        graphics_processes = _parse_graphics_processes(full_output)

    window_minutes = max(1, (policy.recent_xid_window_seconds + 59) // 60)
    journal_output = command(
        "kernel_journal",
        ["journalctl", "-k", "--since", f"-{window_minutes} min", "--no-pager"],
        15,
    )
    if journal_output is not None:
        kernel_events = _filtered_kernel_events(journal_output)

    runtime_output = command("qwen_runtime", _runtime_probe_command(runtime_python), 20)
    if runtime_output is not None:
        try:
            candidate = json.loads(runtime_output.strip().splitlines()[-1])
            if not isinstance(candidate, dict):
                raise ValueError("respuesta no es un objeto")
            runtime = candidate
        except (IndexError, json.JSONDecodeError, ValueError) as exc:
            sources["qwen_runtime"] = f"datos inválidos: {exc}"

    accelerated_apps, worker_pids = _scan_gpu_app_processes(proc_root)
    other_workers = [pid for pid in worker_pids if pid != recognized_worker_pid]
    recent_xid = [line for line in kernel_events if "xid" in line.lower()]
    display_events = [
        line for line in kernel_events if any(term in line.lower() for term in DISPLAY_RISK_TERMS)
    ]

    essential_sources = ["nvidia_smi_gpu", "nvidia_smi_compute", "qwen_runtime"]
    if policy.block_on_recent_xid or policy.block_on_display_engine_warning:
        essential_sources.append("kernel_journal")
    missing = [source for source in essential_sources if sources.get(source) != "ok"]
    if missing:
        message = "No se pudieron obtener datos esenciales: " + ", ".join(missing)
        (blockers if policy.fail_closed else warnings).append(message)

    cuda_available = runtime.get("cuda_available")
    bf16_available = runtime.get("bf16_available")
    if policy.require_cuda and cuda_available is not True:
        blockers.append("CUDA no está disponible en el runtime Qwen")
    if policy.require_bf16 and bf16_available is not True:
        blockers.append("BF16 no está disponible en la GPU del runtime Qwen")

    temperature = metrics.get("temperature_c")
    utilization = metrics.get("gpu_util_percent")
    total = metrics.get("vram_total_mb")
    used = metrics.get("vram_used_mb")
    free = metrics.get("vram_free_mb")
    used_percent = (
        (float(used) / float(total)) * 100
        if isinstance(used, int) and isinstance(total, int) and total > 0
        else None
    )
    if isinstance(temperature, int) and temperature > policy.max_temperature_c:
        blockers.append(
            f"Temperatura GPU {temperature} °C excede el máximo {policy.max_temperature_c} °C"
        )
    if isinstance(utilization, int) and utilization > policy.max_gpu_util_percent:
        blockers.append(
            f"Uso GPU {utilization}% excede el máximo {policy.max_gpu_util_percent}%"
        )
    if isinstance(free, int) and free < policy.min_vram_free_mb:
        blockers.append(
            f"VRAM libre {free} MiB es menor al mínimo {policy.min_vram_free_mb} MiB"
        )
    if used_percent is not None and used_percent > policy.max_vram_used_percent:
        blockers.append(
            f"VRAM usada {used_percent:.1f}% excede el máximo "
            f"{policy.max_vram_used_percent}%"
        )
    if policy.block_on_recent_xid and recent_xid:
        blockers.append(f"Se detectaron {len(recent_xid)} eventos Xid recientes")
    if policy.block_on_display_engine_warning and display_events:
        blockers.append(
            f"Se detectaron {len(display_events)} advertencias recientes del motor gráfico"
        )
    if policy.block_when_worker_exists and other_workers:
        blockers.append(
            "Existe otro worker Qwen no reconocido: " + ", ".join(map(str, other_workers))
        )
    if synthesis_in_progress:
        blockers.append("Ya hay una síntesis Qwen en curso")
    if service_state not in {"offline", "starting", "idle", "blocked"}:
        blockers.append(f"El servicio Qwen está en estado inconsistente: {service_state}")

    session_type = env.get("XDG_SESSION_TYPE")
    display = env.get("DISPLAY")
    if session_type:
        sources["graphical_session"] = "ok"
        if session_type.lower() == "x11":
            warnings.append("La sesión gráfica X11 comparte la GPU NVIDIA con inferencia CUDA")
    else:
        sources["graphical_session"] = "no disponible"
        (blockers if policy.fail_closed else warnings).append(
            "No se pudo determinar el tipo de sesión gráfica"
        )
    if not display:
        warnings.append("DISPLAY no está disponible; no se confirmó la sesión de pantallas")

    if accelerated_apps:
        warnings.append(
            "Aplicaciones gráficas aceleradas detectadas: " + ", ".join(accelerated_apps)
        )
        actions.append("Cierra aplicaciones gráficas pesadas antes de sintetizar")
        if any(app in {"code", "electron"} for app in accelerated_apps):
            actions.append("Abre VS Code para este proyecto con: code --disable-gpu .")
    if blockers:
        actions.insert(0, "No iniciar Qwen hasta resolver todos los bloqueadores")
    actions.append("Qwen comparte la GPU con pantallas; esta mitigación no elimina el riesgo")

    allowed = not blockers
    if not policy.enabled:
        warnings.append("La política GPU fue desactivada explícitamente")
        allowed = not policy.fail_closed
    return GpuPreflightResult(
        allowed=allowed,
        timestamp=timestamp,
        gpu_name=str(metrics["gpu_name"]) if metrics.get("gpu_name") else None,
        driver_version=(
            str(metrics["driver_version"]) if metrics.get("driver_version") else None
        ),
        cuda_version=str(runtime.get("cuda_version") or "") or None,
        temperature_c=temperature if isinstance(temperature, int) else None,
        gpu_util_percent=utilization if isinstance(utilization, int) else None,
        vram_total_mb=total if isinstance(total, int) else None,
        vram_used_mb=used if isinstance(used, int) else None,
        vram_free_mb=free if isinstance(free, int) else None,
        compute_processes=tuple(compute_processes),
        graphics_processes=tuple(graphics_processes),
        recent_kernel_events=tuple(kernel_events),
        recent_xid_events=tuple(recent_xid),
        warnings=tuple(dict.fromkeys(warnings)),
        blockers=tuple(dict.fromkeys(blockers)),
        recommended_actions=tuple(dict.fromkeys(actions)),
        session_type=session_type,
        display=display,
        cuda_available=cuda_available if isinstance(cuda_available, bool) else None,
        bf16_available=bf16_available if isinstance(bf16_available, bool) else None,
        runtime_torch_version=str(runtime.get("torch") or "") or None,
        data_sources=sources,
    )
