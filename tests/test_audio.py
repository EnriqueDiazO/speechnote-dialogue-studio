from __future__ import annotations

import wave
from pathlib import Path

import pytest

from dialogue_studio.audio import (
    MASTER_SAMPLE_RATE,
    concatenate_waves,
    export_mp3,
    has_ffmpeg,
    normalize_audio,
    probe_audio,
    sha256_file,
)


@pytest.mark.skipif(not has_ffmpeg(), reason="FFmpeg no disponible")
def test_ffprobe_and_normalization_preserve_raw(make_wav, tmp_path: Path) -> None:
    raw = make_wav(tmp_path / "raw.wav", rate=24_000, channels=2)
    before = sha256_file(raw)
    normalized = tmp_path / "normalized.wav"
    source_info = probe_audio(raw)
    assert source_info.sample_rate == 24_000
    assert source_info.channels == 2
    info = normalize_audio(raw, normalized)
    assert info.is_master_format
    assert sha256_file(raw) == before
    assert sha256_file(normalized) != before


def test_concatenation_adds_exact_pause_and_hash(make_wav, tmp_path: Path) -> None:
    segments = [make_wav(tmp_path / f"{index}.wav", duration=0.1) for index in range(3)]
    master = tmp_path / "master.wav"
    info = concatenate_waves(segments, master, 650)
    assert info.duration_seconds == pytest.approx(1.6, abs=1 / MASTER_SAMPLE_RATE)
    with wave.open(str(master), "rb") as audio:
        assert audio.getframerate() == 48_000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getnframes() == 76_800
    assert len(sha256_file(master)) == 64


def test_concatenation_rejects_wrong_format(make_wav, tmp_path: Path) -> None:
    segment = make_wav(tmp_path / "wrong.wav", rate=24_000)
    with pytest.raises(RuntimeError, match="normalizado"):
        concatenate_waves([segment], tmp_path / "master.wav", 650)


@pytest.mark.skipif(not has_ffmpeg(), reason="FFmpeg no disponible")
def test_mp3_is_optional_and_uses_expected_audio_shape(make_wav, tmp_path: Path) -> None:
    source = make_wav(tmp_path / "master.wav", duration=0.2)
    destination = tmp_path / "master.mp3"
    info = export_mp3(source, destination)
    assert destination.is_file()
    assert info.codec == "mp3"
    assert info.sample_rate == 48_000
    assert info.channels == 1
