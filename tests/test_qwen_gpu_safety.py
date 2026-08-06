from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dialogue_studio.qwen_gpu_safety import (
    GpuPreflightResult,
    GpuSafetyPolicy,
    load_gpu_safety_policy,
)
from dialogue_studio.qwen_preflight import run_gpu_preflight


def test_gpu_policy_uses_conservative_editable_defaults(tmp_path) -> None:
    policy = GpuSafetyPolicy()
    assert policy.enabled is policy.fail_closed is True
    assert policy.max_gpu_util_percent == 75
    assert policy.min_vram_free_mb == 3000
    assert policy.allow_cpu_fallback is False

    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps({**policy.to_dict(), "max_temperature_c": 70}), encoding="utf-8"
    )
    assert load_gpu_safety_policy(path).max_temperature_c == 70


@pytest.mark.parametrize(
    "changes",
    (
        {"max_gpu_util_percent": 101},
        {"min_vram_free_mb": -1},
        {"monitor_interval_seconds": 0},
        {"idle_unload_seconds": 301, "idle_shutdown_seconds": 300},
        {"unknown": True},
    ),
)
def test_gpu_policy_rejects_invalid_values(changes) -> None:
    with pytest.raises(ValueError):
        GpuSafetyPolicy.from_mapping({**GpuSafetyPolicy().to_dict(), **changes})


def test_preflight_result_serializes_nested_processes() -> None:
    result = GpuPreflightResult.blocked("CUDA no está disponible")
    assert result.to_dict()["allowed"] is False
    assert result.to_dict()["blockers"] == ("CUDA no está disponible",)


class FakeRunner:
    def __init__(
        self,
        *,
        gpu: str = "Fake RTX, 535.1, 45, 10, 8192, 1024, 7168",
        cuda: bool = True,
        bf16: bool = True,
        journal: str = "",
        journal_returncode: int = 0,
    ) -> None:
        self.gpu = gpu
        self.cuda = cuda
        self.bf16 = bf16
        self.journal = journal
        self.journal_returncode = journal_returncode

    def __call__(self, command, _timeout):
        if "--query-gpu" in " ".join(command):
            return subprocess.CompletedProcess(command, 0, self.gpu + "\n", "")
        if "--query-compute-apps" in " ".join(command):
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, "Processes:\n", "")
        if command[0] == "journalctl":
            return subprocess.CompletedProcess(
                command,
                self.journal_returncode,
                self.journal,
                "No journal files" if self.journal_returncode else "",
            )
        payload = json.dumps(
            {
                "torch": "2.7.1+cu118",
                "cuda_version": "11.8",
                "cuda_available": self.cuda,
                "bf16_available": self.bf16,
            }
        )
        return subprocess.CompletedProcess(command, 0, payload + "\n", "")


def preflight(runner: FakeRunner, tmp_path: Path, **kwargs):
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    return run_gpu_preflight(
        kwargs.pop("policy", GpuSafetyPolicy()),
        Path("/qwen/python"),
        runner=runner,
        proc_root=proc,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"},
        **kwargs,
    )


def test_safe_preflight_allows_with_expected_x11_warning(tmp_path) -> None:
    result = preflight(FakeRunner(), tmp_path)
    assert result.allowed is True
    assert result.gpu_name == "Fake RTX"
    assert result.vram_free_mb == 7168
    assert result.warnings == (
        "La sesión gráfica X11 comparte la GPU NVIDIA con inferencia CUDA",
    )


@pytest.mark.parametrize(
    ("runner", "message"),
    (
        (FakeRunner(cuda=False), "CUDA no está disponible"),
        (FakeRunner(bf16=False), "BF16 no está disponible"),
        (
            FakeRunner(gpu="Fake RTX, 535.1, 45, 10, 8192, 6000, 2192"),
            "VRAM libre",
        ),
        (FakeRunner(gpu="Fake RTX, 535.1, 80, 10, 8192, 1024, 7168"), "Temperatura"),
        (FakeRunner(journal="kernel: NVRM: Xid 31, pid=9\n"), "eventos Xid"),
    ),
)
def test_preflight_blocks_known_gpu_risks(runner, message, tmp_path) -> None:
    result = preflight(runner, tmp_path)
    assert result.allowed is False
    assert any(message in blocker for blocker in result.blockers)


def test_journal_permission_failure_is_fail_closed_by_default(tmp_path) -> None:
    result = preflight(FakeRunner(journal_returncode=1), tmp_path)
    assert result.allowed is False
    assert any("kernel_journal" in blocker for blocker in result.blockers)


def test_journal_permission_failure_can_only_fail_open_when_explicit(tmp_path) -> None:
    policy = GpuSafetyPolicy(fail_closed=False)
    result = preflight(FakeRunner(journal_returncode=1), tmp_path, policy=policy)
    assert result.allowed is True
    assert any("kernel_journal" in warning for warning in result.warnings)


def test_preflight_blocks_unrecognized_worker_and_active_synthesis(tmp_path) -> None:
    proc = tmp_path / "proc"
    worker = proc / "42"
    worker.mkdir(parents=True)
    (worker / "cmdline").write_bytes(b"python\0-m\0dialogue_studio.qwen_worker\0")
    result = run_gpu_preflight(
        GpuSafetyPolicy(),
        Path("/qwen/python"),
        runner=FakeRunner(),
        proc_root=proc,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"},
        synthesis_in_progress=True,
    )
    assert result.allowed is False
    assert any("worker Qwen" in blocker for blocker in result.blockers)
    assert any("síntesis Qwen" in blocker for blocker in result.blockers)
