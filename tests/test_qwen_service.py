from __future__ import annotations

import contextlib
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from dialogue_studio.qwen_service import (
    FALLBACK_LANGUAGES,
    FALLBACK_SPEAKERS,
    QwenRuntime,
    QwenServiceConfig,
    QwenServiceError,
    validate_generation_options,
)


class FakeCuda:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.emptied = 0

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported() -> bool:
        return True

    @staticmethod
    def get_device_name(_index: int) -> str:
        return "Fake RTX"

    @staticmethod
    def mem_get_info(_index: int) -> tuple[int, int]:
        return 4_000, 8_000

    def manual_seed_all(self, seed: int) -> None:
        self.seeds.append(seed)

    def empty_cache(self) -> None:
        self.emptied += 1


class FakeRandom:
    def __init__(self) -> None:
        self.entries = 0
        self.exits = 0

    @contextlib.contextmanager
    def fork_rng(self, **_kwargs):
        self.entries += 1
        try:
            yield
        finally:
            self.exits += 1


class FakeTorch:
    __version__ = "2.7.1"
    version = SimpleNamespace(cuda="11.8")
    bfloat16 = object()

    def __init__(self) -> None:
        self.cuda = FakeCuda()
        self.random = FakeRandom()
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)


class FakeModel:
    def __init__(self) -> None:
        self.model = SimpleNamespace(tts_model_size="0b6")
        self.calls: list[dict[str, object]] = []

    @staticmethod
    def get_supported_speakers() -> list[str]:
        return ["Serena", "Vivian", "Ryan"]

    @staticmethod
    def get_supported_languages() -> list[str]:
        return ["Spanish", "English"]

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        return [[0.0] * 240], 24_000


def config(tmp_path: Path) -> QwenServiceConfig:
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
    )


def write_wave(path: Path, waveform, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * len(waveform))


def test_runtime_loads_once_forwards_supported_kwargs_and_restores_rng(tmp_path: Path) -> None:
    fake_torch = FakeTorch()
    fake_model = FakeModel()
    loads = []
    runtime = QwenRuntime(
        config(tmp_path),
        torch_loader=lambda: fake_torch,
        model_loader=lambda _config, _torch: loads.append(True) or fake_model,
        wave_writer=write_wave,
        finite_checker=lambda _waveform: True,
    )
    runtime.initialize()
    before = runtime.capabilities()
    assert before["source"] == "verified_fallback"
    assert before["speakers"] == list(FALLBACK_SPEAKERS)
    assert before["languages"] == list(FALLBACK_LANGUAGES)
    assert before["supports_instruct"] is False
    assert before["supports_sampling_controls"] is True
    assert before["supports_voice_design"] is False
    assert before["supports_voice_cloning"] is False

    expected_options = {
        "seed": 7,
        "max_new_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }
    for voice in ("serena", "vivian"):
        output = tmp_path / "projects" / f"{voice}.wav"
        result = runtime.synthesize(
            {
                "text": "Texto de prueba",
                "speaker": voice,
                "language": "spanish",
                "output_path": str(output),
                "generation_options": expected_options,
            }
        )
        assert result["sample_rate"] == 24_000
        assert output.read_bytes()[:4] == b"RIFF"
    assert len(loads) == 1
    assert runtime.health()["load_count"] == 1
    assert runtime.capabilities()["speakers"] == ["serena", "vivian", "ryan"]
    assert set(fake_model.calls[0]) == {
        "text",
        "speaker",
        "language",
        "non_streaming_mode",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
    }
    assert fake_model.calls[0]["temperature"] == 0.8
    assert fake_torch.seeds == [7, 7]
    assert fake_torch.random.entries == fake_torch.random.exits == 2
    assert not list(tmp_path.rglob("*.partial"))


def test_runtime_rejects_instruct_for_installed_model(tmp_path: Path) -> None:
    runtime = QwenRuntime(
        config(tmp_path),
        torch_loader=FakeTorch,
        model_loader=lambda _config, _torch: FakeModel(),
        wave_writer=write_wave,
        finite_checker=lambda _waveform: True,
    )
    runtime.initialize()
    with pytest.raises(QwenServiceError) as error:
        runtime.synthesize(
            {
                "text": "Hola",
                "speaker": "serena",
                "language": "spanish",
                "instruct": "Tono cálido",
                "output_path": str(tmp_path / "audio" / "one.wav"),
            }
        )
    assert error.value.code == "unsupported_instruct"


def test_runtime_releases_gpu_lock_after_failure_and_preserves_partial(tmp_path: Path) -> None:
    fake_model = FakeModel()
    fail_once = [True]

    def writer(path: Path, waveform, sample_rate: int) -> None:
        path.write_bytes(b"partial")
        if fail_once and fail_once.pop():
            raise RuntimeError("write failed")
        write_wave(path, waveform, sample_rate)

    runtime = QwenRuntime(
        config(tmp_path),
        torch_loader=FakeTorch,
        model_loader=lambda _config, _torch: fake_model,
        wave_writer=writer,
        finite_checker=lambda _waveform: True,
    )
    runtime.initialize()
    payload = {
        "text": "Hola",
        "speaker": "serena",
        "language": "spanish",
        "output_path": str(tmp_path / "audio" / "one.wav"),
    }
    with pytest.raises(QwenServiceError, match="write failed"):
        runtime.synthesize(payload)
    assert list(tmp_path.rglob("*.partial"))
    payload["output_path"] = str(tmp_path / "audio" / "two.wav")
    assert runtime.synthesize(payload)["ok"] is True


def test_runtime_rejects_concurrency_unsafe_paths_and_unsupported_options(
    tmp_path: Path,
) -> None:
    runtime = QwenRuntime(config(tmp_path), torch_loader=FakeTorch)
    assert runtime._generation_lock.acquire(blocking=False)
    try:
        with pytest.raises(QwenServiceError) as busy:
            runtime.synthesize(
                {
                    "text": "Hola",
                    "speaker": "serena",
                    "language": "spanish",
                    "output_path": str(tmp_path / "x.wav"),
                }
            )
        assert busy.value.code == "gpu_busy"
    finally:
        runtime._generation_lock.release()

    with pytest.raises(QwenServiceError, match="fuera"):
        runtime.synthesize(
            {
                "text": "Hola",
                "speaker": "serena",
                "language": "spanish",
                "output_path": str(tmp_path.parent / "outside.wav"),
            }
        )
    with pytest.raises(QwenServiceError, match="no soportadas"):
        validate_generation_options({"fake": 1})
    with pytest.raises(QwenServiceError, match="temperature"):
        validate_generation_options({"temperature": 9.0})


def test_unload_refuses_while_generating_and_releases_vram(tmp_path: Path) -> None:
    fake_torch = FakeTorch()
    runtime = QwenRuntime(
        config(tmp_path),
        torch_loader=lambda: fake_torch,
        model_loader=lambda _config, _torch: FakeModel(),
        wave_writer=write_wave,
        finite_checker=lambda _waveform: True,
    )
    runtime.initialize()
    runtime._ensure_model()
    assert runtime._generation_lock.acquire(blocking=False)
    try:
        with pytest.raises(QwenServiceError) as busy:
            runtime.unload()
        assert busy.value.code == "gpu_busy"
    finally:
        runtime._generation_lock.release()
    assert runtime.unload() == {"ok": True, "unloaded": True}
    assert fake_torch.cuda.emptied == 1


def test_threaded_busy_request_does_not_wait(tmp_path: Path) -> None:
    runtime = QwenRuntime(config(tmp_path), torch_loader=FakeTorch)
    held = threading.Event()
    release = threading.Event()

    def owner() -> None:
        runtime._generation_lock.acquire()
        held.set()
        release.wait(timeout=2)
        runtime._generation_lock.release()

    thread = threading.Thread(target=owner)
    thread.start()
    assert held.wait(timeout=1)
    try:
        with pytest.raises(QwenServiceError) as error:
            runtime.synthesize(
                {
                    "text": "Hola",
                    "speaker": "serena",
                    "language": "spanish",
                    "output_path": str(tmp_path / "x.wav"),
                }
            )
        assert error.value.retryable is True
    finally:
        release.set()
        thread.join(timeout=1)
