from __future__ import annotations

import io
import json
import queue
import threading
import time
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


def safe_metrics() -> dict[str, object]:
    return {
        "gpu_name": "Fake RTX",
        "driver_version": "535.1",
        "temperature_c": 45,
        "gpu_util_percent": 20,
        "vram_total_mb": 8192,
        "vram_used_mb": 2048,
        "vram_free_mb": 6144,
    }


def no_kernel_events(_window: int) -> tuple[str, ...]:
    return ()


def test_controller_spawns_exactly_one_managed_worker_without_fork(tmp_path) -> None:
    calls = []
    process = FakeProcess()

    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return process

    controller = QwenController(
        config(tmp_path),
        popen=popen,
        preflight_runner=safe_preflight,
        metric_sampler=safe_metrics,
        kernel_sampler=no_kernel_events,
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
        metric_sampler=safe_metrics,
        kernel_sampler=no_kernel_events,
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


class QueueOutput:
    def __init__(self) -> None:
        self.lines: queue.Queue[str | None] = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self.lines.get(timeout=2)
        if line is None:
            raise StopIteration
        return line

    def send(self, payload: dict[str, object]) -> None:
        self.lines.put(json.dumps(payload) + "\n")

    def close(self) -> None:
        self.lines.put(None)


class SimulatedWorkerProcess:
    next_pid = 9000

    def __init__(self, *, release: threading.Event | None = None) -> None:
        self.pid = SimulatedWorkerProcess.next_pid
        SimulatedWorkerProcess.next_pid += 1
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.stdout = QueueOutput()
        self.stdin = self
        self.release = release
        self.started = threading.Event()
        self.active_syntheses = 0
        self.max_active_syntheses = 0
        self.synthesis_count = 0
        self.stdout.send(
            {
                "event": "ready",
                "pid": self.pid,
                "health": {"model_loaded": False, "state": "idle"},
            }
        )

    def write(self, line: str) -> int:
        command = json.loads(line)
        request_id = command["request_id"]
        if command["command"] == "synthesize":
            self.active_syntheses += 1
            self.max_active_syntheses = max(
                self.max_active_syntheses, self.active_syntheses
            )
            self.synthesis_count += 1
            self.started.set()
            self.stdout.send(
                {"event": "status", "state": "generating", "request_id": request_id}
            )
            def finish() -> None:
                if self.release is not None:
                    self.release.wait(timeout=2)
                else:
                    time.sleep(0.01)
                self.stdout.send(
                    {
                        "event": "response",
                        "request_id": request_id,
                        "result": {"ok": True, "sequence": self.synthesis_count},
                    }
                )
                self.active_syntheses -= 1

            if self.release is None:
                finish()
            else:
                threading.Thread(target=finish, daemon=True).start()
        elif command["command"] == "unload":
            self.stdout.send(
                {"event": "response", "request_id": request_id, "result": {"ok": True}}
            )
        return len(line)

    @staticmethod
    def flush() -> None:
        return

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self.stdout.close()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdout.close()

    def wait(self, timeout=None):
        return self.returncode


def test_concurrent_requests_use_one_worker_and_one_synthesis_at_a_time(tmp_path) -> None:
    processes = []

    def popen(*_args, **_kwargs):
        process = SimulatedWorkerProcess()
        processes.append(process)
        return process

    controller = QwenController(
        config(tmp_path),
        popen=popen,
        preflight_runner=safe_preflight,
        metric_sampler=safe_metrics,
        kernel_sampler=no_kernel_events,
    )
    controller.start()
    results = []

    def submit(index: int) -> None:
        results.append(controller.submit({"text": str(index)}))

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(6)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
        assert len(results) == 6
        assert len(processes) == 1
        assert processes[0].synthesis_count == 6
        assert processes[0].max_active_syntheses == 1
        assert not controller.health()["queue"]
    finally:
        controller.close()


def test_cancel_removes_pending_jobs_without_canceling_current_when_requested(tmp_path) -> None:
    release = threading.Event()
    process = SimulatedWorkerProcess(release=release)
    controller = QwenController(
        config(tmp_path),
        popen=lambda *_args, **_kwargs: process,
        preflight_runner=safe_preflight,
        metric_sampler=safe_metrics,
        kernel_sampler=no_kernel_events,
    )
    controller.start()
    outcomes = []

    def submit(index: int) -> None:
        try:
            outcomes.append((index, controller.submit({"text": str(index)})["ok"]))
        except QwenServiceError as exc:
            outcomes.append((index, exc.code))

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(3)]
    try:
        for thread in threads:
            thread.start()
        assert process.started.wait(timeout=1)
        canceled = controller.cancel(stop_current=False)
        assert canceled["canceled_pending"] == 2
        assert canceled["current_cancel_requested"] is False
        release.set()
        for thread in threads:
            thread.join(timeout=2)
        values = [value for _, value in outcomes]
        assert values.count("canceled") == 2
        assert values.count(True) == 1
    finally:
        release.set()
        controller.close()


def test_new_xid_terminates_worker_and_latches_gpu_fault(tmp_path) -> None:
    release = threading.Event()
    process = SimulatedWorkerProcess(release=release)
    controller = QwenController(
        config(
            tmp_path,
            monitor_interval_seconds=0.01,
            kernel_poll_interval_seconds=0.01,
        ),
        popen=lambda *_args, **_kwargs: process,
        preflight_runner=safe_preflight,
        metric_sampler=safe_metrics,
        kernel_sampler=lambda _window: ("kernel: NVRM: Xid 31, pid=9000",),
    )
    controller.start()
    outcome = []

    def submit() -> None:
        try:
            controller.submit({"text": "fault"})
        except QwenServiceError as exc:
            outcome.append(exc.code)

    thread = threading.Thread(target=submit)
    try:
        thread.start()
        thread.join(timeout=2)
        assert outcome == ["gpu_fault"]
        assert process.terminated is True
        assert controller.health()["state"] == "gpu_fault"
        assert "Xid nuevo" in str(controller.health()["gpu_fault_reason"])
    finally:
        release.set()
        controller.close()


def test_critical_temperature_stops_current_queue(tmp_path) -> None:
    process = SimulatedWorkerProcess()
    hot_metrics = {**safe_metrics(), "temperature_c": 80}
    controller = QwenController(
        config(tmp_path),
        popen=lambda *_args, **_kwargs: process,
        preflight_runner=safe_preflight,
        metric_sampler=lambda: hot_metrics,
        kernel_sampler=no_kernel_events,
    )
    controller._start_worker()
    try:
        blocker = controller._monitor_worker("generating", poll_kernel=False)
        assert blocker is not None
        assert "Temperatura GPU crítica" in blocker
    finally:
        controller.close()


def test_idle_unload_then_shutdown_releases_worker(tmp_path) -> None:
    process = SimulatedWorkerProcess()
    controller = QwenController(
        config(tmp_path, idle_unload_seconds=120, idle_shutdown_seconds=300),
        popen=lambda *_args, **_kwargs: process,
        preflight_runner=safe_preflight,
        metric_sampler=safe_metrics,
        kernel_sampler=no_kernel_events,
    )
    controller._start_worker()
    controller._model_loaded = True
    baseline = controller._last_activity_monotonic
    try:
        assert controller._lifecycle_tick(baseline + 121) == "idle_unload"
        assert controller.health()["model_loaded"] is False
        assert controller.health()["last_unload"] == {"ok": True}
        assert controller._lifecycle_tick(baseline + 301) == "idle_shutdown"
        assert process.terminated is True
        assert controller.health()["worker_alive"] is False
    finally:
        controller.close()


def test_controller_restarts_worker_after_controlled_stop(tmp_path) -> None:
    processes = [SimulatedWorkerProcess(), SimulatedWorkerProcess()]
    controller = QwenController(
        config(tmp_path),
        popen=lambda *_args, **_kwargs: processes.pop(0),
        preflight_runner=safe_preflight,
        metric_sampler=safe_metrics,
        kernel_sampler=no_kernel_events,
    )
    controller._start_worker()
    first_pid = controller.health()["worker_pid"]
    controller.stop_worker()
    controller._start_worker()
    try:
        assert controller.health()["worker_pid"] != first_pid
        assert controller.health()["worker_alive"] is True
    finally:
        controller.close()
