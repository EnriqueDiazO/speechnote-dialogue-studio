from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from dialogue_studio.qwen_gpu_safety import GpuPreflightResult, GpuSafetyPolicy
from dialogue_studio.qwen_service import QwenController, QwenServiceConfig, QwenServiceError


def config(tmp_path: Path, **policy_changes) -> QwenServiceConfig:
    policy = GpuSafetyPolicy.from_mapping(
        {**GpuSafetyPolicy().to_dict(), "post_job_cooldown_seconds": 0, **policy_changes}
    )
    return QwenServiceConfig(
        model="test-model",
        host="127.0.0.1",
        port=8765,
        device="cuda:0",
        dtype="bfloat16",
        attention="sdpa",
        timeout=10,
        output_root=tmp_path,
        pid_file=tmp_path / "runtime" / "qwen.pid",
        policy=policy,
    )


class FakeProcess:
    next_pid = 8100

    def __init__(self, *, ready: bool = True) -> None:
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        health = {"model_loaded": False, "state": "idle"}
        line = json.dumps({"event": "ready", "pid": self.pid, "health": health}) + "\n"
        self.stdout = io.StringIO(line if ready else "")
        self.stdin = io.StringIO()
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def safe_preflight(*_args, **_kwargs) -> GpuPreflightResult:
    return GpuPreflightResult(allowed=True, timestamp="2026-08-05T00:00:00Z")


def test_controller_spawns_exactly_one_managed_worker_without_fork(tmp_path) -> None:
    calls = []
    process = FakeProcess()

    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return process

    controller = QwenController(
        config(tmp_path), popen=popen, preflight_runner=safe_preflight
    )
    controller.start()
    try:
        controller._start_worker()
        controller._start_worker()
        assert len(calls) == 1
        assert calls[0][0][-2:] == ["-m", "dialogue_studio.qwen_worker"]
        assert calls[0][1]["start_new_session"] is True
        assert controller.health()["worker_creation_method"] == "subprocess_spawn"
        assert controller.health()["worker_pid"] == process.pid
    finally:
        controller.close()
    assert process.terminated is True


def test_controller_start_timeout_only_terminates_its_worker(tmp_path) -> None:
    process = FakeProcess(ready=False)
    controller = QwenController(
        config(tmp_path, worker_start_timeout_seconds=0),
        popen=lambda *_args, **_kwargs: process,
        preflight_runner=safe_preflight,
    )
    with pytest.raises(QwenServiceError) as error:
        controller._start_worker()
    assert error.value.code == "worker_start_timeout"
    assert process.terminated is True
    assert process.killed is False


def test_controller_source_never_imports_torch_or_qwen() -> None:
    source = Path("dialogue_studio/qwen_service.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from qwen_tts" not in source
