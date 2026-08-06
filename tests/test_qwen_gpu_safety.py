from __future__ import annotations

import json

import pytest

from dialogue_studio.qwen_gpu_safety import (
    GpuPreflightResult,
    GpuSafetyPolicy,
    load_gpu_safety_policy,
)


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
