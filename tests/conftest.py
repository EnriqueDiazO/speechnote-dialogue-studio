from __future__ import annotations

import wave
from pathlib import Path

import pytest


@pytest.fixture
def make_wav():
    def _make(
        path: Path,
        *,
        duration: float = 0.1,
        rate: int = 48_000,
        channels: int = 1,
        width: int = 2,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = round(duration * rate)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(width)
            output.setframerate(rate)
            output.writeframes(b"\x00" * frames * channels * width)
        return path

    return _make
