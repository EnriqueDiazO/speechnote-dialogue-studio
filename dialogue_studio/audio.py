"""Audio inspection, normalization and deterministic master generation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MASTER_CODEC = "pcm_s16le"
MASTER_SAMPLE_RATE = 48_000
MASTER_CHANNELS = 1
MASTER_SAMPLE_WIDTH = 2


class AudioError(RuntimeError):
    """Audio processing failed."""


@dataclass(frozen=True)
class AudioInfo:
    codec: str
    sample_rate: int
    channels: int
    duration_seconds: float

    @property
    def is_master_format(self) -> bool:
        return (
            self.codec == MASTER_CODEC
            and self.sample_rate == MASTER_SAMPLE_RATE
            and self.channels == MASTER_CHANNELS
        )


Runner = Callable[..., subprocess.CompletedProcess[str]]


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _clean_stderr(value: str, limit: int = 800) -> str:
    return " ".join(value.strip().split())[-limit:]


def probe_audio(path: Path, *, runner: Runner = subprocess.run) -> AudioInfo:
    try:
        result = runner(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioError("ffprobe no está disponible") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError("ffprobe excedió el tiempo límite") from exc
    if result.returncode != 0:
        raise AudioError(f"ffprobe no pudo leer el audio: {_clean_stderr(result.stderr)}")
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        duration = float(data["format"]["duration"])
        return AudioInfo(
            codec=str(stream["codec_name"]),
            sample_rate=int(stream["sample_rate"]),
            channels=int(stream["channels"]),
            duration_seconds=duration,
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AudioError("ffprobe devolvió datos de audio incompletos") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise AudioError("No se sobrescriben enlaces simbólicos")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(suffix=destination.suffix, dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_audio(
    source: Path,
    destination: Path,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 120,
) -> AudioInfo:
    source_info = probe_audio(source, runner=runner)
    if source_info.is_master_format:
        _atomic_copy(source, destination)
        return probe_audio(destination, runner=runner)
    if destination.is_symlink():
        raise AudioError("No se sobrescriben enlaces simbólicos")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(suffix=".wav", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        result = runner(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-ar",
                str(MASTER_SAMPLE_RATE),
                "-ac",
                str(MASTER_CHANNELS),
                "-c:a",
                MASTER_CODEC,
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise AudioError(f"FFmpeg no pudo normalizar: {_clean_stderr(result.stderr)}")
        info = probe_audio(temporary, runner=runner)
        if not info.is_master_format:
            raise AudioError("FFmpeg no produjo el formato maestro esperado")
        os.replace(temporary, destination)
        return info
    except FileNotFoundError as exc:
        raise AudioError("FFmpeg no está disponible") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError("FFmpeg excedió el tiempo límite al normalizar") from exc
    finally:
        temporary.unlink(missing_ok=True)


def concatenate_waves(
    segments: list[Path],
    destination: Path,
    pause_ms: int,
) -> AudioInfo:
    if not segments:
        raise ValueError("Se necesita al menos una intervención para construir el diálogo")
    if not 0 <= pause_ms <= 5000:
        raise ValueError("pause_ms debe estar entre 0 y 5000")
    if destination.is_symlink():
        raise AudioError("No se sobrescriben enlaces simbólicos")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(suffix=".wav", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    total_frames = 0
    try:
        with wave.open(str(temporary), "wb") as output:
            output.setnchannels(MASTER_CHANNELS)
            output.setsampwidth(MASTER_SAMPLE_WIDTH)
            output.setframerate(MASTER_SAMPLE_RATE)
            for index, segment in enumerate(segments):
                with wave.open(str(segment), "rb") as source:
                    params = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                    expected = (MASTER_CHANNELS, MASTER_SAMPLE_WIDTH, MASTER_SAMPLE_RATE)
                    if params != expected or source.getcomptype() != "NONE":
                        raise AudioError(f"El segmento no está normalizado: {segment.name}")
                    frame_count = source.getnframes()
                    output.writeframes(source.readframes(frame_count))
                    total_frames += frame_count
                if index < len(segments) - 1 and pause_ms:
                    silence_frames = round(MASTER_SAMPLE_RATE * pause_ms / 1000)
                    output.writeframes(b"\x00" * silence_frames * MASTER_SAMPLE_WIDTH)
                    total_frames += silence_frames
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return AudioInfo(MASTER_CODEC, MASTER_SAMPLE_RATE, MASTER_CHANNELS, total_frames / 48_000)


def export_mp3(
    source_wav: Path,
    destination: Path,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 180,
) -> AudioInfo:
    if destination.is_symlink():
        raise AudioError("No se sobrescriben enlaces simbólicos")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(suffix=".mp3", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        result = runner(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source_wav),
                "-ar",
                str(MASTER_SAMPLE_RATE),
                "-ac",
                "1",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise AudioError(f"FFmpeg no pudo crear el MP3: {_clean_stderr(result.stderr)}")
        os.replace(temporary, destination)
        return probe_audio(destination, runner=runner)
    except FileNotFoundError as exc:
        raise AudioError("FFmpeg no está disponible; el WAV sigue operativo") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioError("FFmpeg excedió el tiempo límite al crear el MP3") from exc
    finally:
        temporary.unlink(missing_ok=True)
