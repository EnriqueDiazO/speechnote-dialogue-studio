"""Safe adapter for Speech Note's Flatpak command-line interface."""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

APP_ID = "net.mkiol.SpeechNote"
MODEL_PATTERN = re.compile(r'^\s*([\w.-]+)\s+"([^"]+)"\s*$')


class SpeechNoteError(RuntimeError):
    """A user-actionable Speech Note error."""


@dataclass(frozen=True)
class TTSModel:
    model_id: str
    label: str


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[..., subprocess.CompletedProcess[str]]
_SYNTHESIS_LOCK = threading.Lock()


def _run(
    arguments: Sequence[str],
    *,
    timeout: float = 30,
    runner: Runner = subprocess.run,
) -> CommandResult:
    try:
        result = runner(
            list(arguments),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SpeechNoteError("Flatpak no está instalado") from exc
    except subprocess.TimeoutExpired as exc:
        raise SpeechNoteError("Speech Note no respondió antes del timeout") from exc
    return CommandResult(result.returncode, result.stdout or "", result.stderr or "")


def _flatpak_command(*arguments: str) -> list[str]:
    return ["flatpak", "run", APP_ID, *arguments]


def check_flatpak(*, runner: Runner = subprocess.run) -> bool:
    if runner is subprocess.run and shutil.which("flatpak") is None:
        return False
    try:
        return _run(["flatpak", "--version"], timeout=5, runner=runner).returncode == 0
    except SpeechNoteError:
        return False


def check_speechnote_installed(*, runner: Runner = subprocess.run) -> bool:
    if not check_flatpak(runner=runner):
        return False
    return _run(["flatpak", "info", APP_ID], timeout=10, runner=runner).returncode == 0


def check_speechnote_open(*, runner: Runner = subprocess.run) -> bool:
    result = _run(["flatpak", "ps", "--columns=application"], timeout=5, runner=runner)
    return result.returncode == 0 and APP_ID in result.stdout.split()


def parse_tts_models(output: str) -> list[TTSModel]:
    models: list[TTSModel] = []
    for line in output.splitlines():
        match = MODEL_PATTERN.match(line)
        if match:
            models.append(TTSModel(match.group(1), match.group(2)))
    return models


def list_tts_models(*, runner: Runner = subprocess.run) -> list[TTSModel]:
    result = _run(
        _flatpak_command("--print-available-models", "tts"),
        timeout=30,
        runner=runner,
    )
    if result.returncode != 0:
        raise _friendly_error(result.stderr or result.stdout)
    return parse_tts_models(result.stdout)


def get_active_tts_model(*, runner: Runner = subprocess.run) -> TTSModel | None:
    result = _run(_flatpak_command("--print-active-model", "tts"), runner=runner)
    if result.returncode != 0:
        raise _friendly_error(result.stderr or result.stdout)
    models = parse_tts_models(result.stdout)
    return models[0] if models else None


def set_tts_model(model_id: str, *, runner: Runner = subprocess.run) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_id):
        raise ValueError("Identificador de voz no válido")
    result = _run(
        _flatpak_command("--action", "set-tts-model", "--id", model_id),
        runner=runner,
    )
    if result.returncode != 0:
        raise _friendly_error(result.stderr or result.stdout)


def _friendly_error(details: str) -> SpeechNoteError:
    clean = " ".join(details.strip().split())
    lowered = clean.lower()
    if "action invocation is not enabled" in lowered:
        return SpeechNoteError(
            "La invocación externa está deshabilitada. En Speech Note abre Ajustes → "
            "Permitir aplicaciones externas para invocar acciones."
        )
    if "not installed" in lowered or "is not installed" in lowered:
        return SpeechNoteError("Speech Note no está instalado")
    if "not running" in lowered or "connection" in lowered or "dbus" in lowered:
        return SpeechNoteError("Speech Note no está abierto o su sesión gráfica no responde")
    if "model" in lowered and ("not found" in lowered or "unknown" in lowered):
        return SpeechNoteError("El modelo de voz no está disponible")
    if "model" in lowered and ("download" in lowered or "not installed" in lowered):
        return SpeechNoteError("El modelo de voz no está descargado")
    if "permission" in lowered or "denied" in lowered:
        return SpeechNoteError("Flatpak no tiene acceso a la ruta de salida")
    return SpeechNoteError(f"Speech Note devolvió un error: {clean or 'sin detalles'}")


def _assert_controlled_output(output_path: Path, controlled_root: Path) -> None:
    root = controlled_root.resolve()
    output = output_path.resolve(strict=False)
    if root != output and root not in output.parents:
        raise ValueError("La salida debe estar dentro de la carpeta controlada")
    cursor = output
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError("No se permiten enlaces simbólicos como destino")
        cursor = cursor.parent


def _wave_header_is_valid(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
    except OSError:
        return False
    return len(header) == 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"


def wait_for_wave(
    path: Path,
    *,
    timeout: float = 300,
    stable_seconds: float = 1.5,
    poll_interval: float = 0.25,
    probe: Callable[[Path], object] | None = None,
) -> None:
    started = time.monotonic()
    stable_since: float | None = None
    last_size = -1
    while time.monotonic() - started < timeout:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = -1
        now = time.monotonic()
        if size > 44:
            if size == last_size:
                stable_since = stable_since or now
                if now - stable_since >= stable_seconds:
                    if not _wave_header_is_valid(path):
                        raise SpeechNoteError("Speech Note produjo un archivo WAV inválido")
                    if probe is not None:
                        try:
                            probe(path)
                        except (OSError, RuntimeError, ValueError) as exc:
                            raise SpeechNoteError(
                                "ffprobe rechazó el archivo WAV generado"
                            ) from exc
                    return
            else:
                stable_since = now
        else:
            stable_since = None
        last_size = size
        time.sleep(poll_interval)
    if not path.exists():
        raise SpeechNoteError("Speech Note no creó el archivo de salida antes del timeout")
    raise SpeechNoteError("Timeout esperando que Speech Note terminara el archivo WAV")


def synthesize_text(
    model_id: str,
    text: str,
    output_path: Path,
    controlled_root: Path,
    *,
    command_timeout: float = 30,
    output_timeout: float = 300,
    runner: Runner = subprocess.run,
    probe: Callable[[Path], object] | None = None,
) -> None:
    if not text.strip():
        raise ValueError("No se puede sintetizar texto vacío")
    if not model_id.strip() or not re.fullmatch(r"[A-Za-z0-9_.-]+", model_id):
        raise ValueError("La intervención necesita una voz válida")
    _assert_controlled_output(output_path, controlled_root)
    if output_path.exists():
        raise FileExistsError("La salida ya existe; usa una ruta nueva para regenerar")
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not _SYNTHESIS_LOCK.acquire(blocking=False):
        raise SpeechNoteError("Ya hay una síntesis en curso")
    try:
        result = _run(
            _flatpak_command(
                "--action",
                "start-reading-text",
                "--id",
                model_id,
                "--text",
                text,
                "--output-file",
                str(output_path),
            ),
            timeout=command_timeout,
            runner=runner,
        )
        if result.returncode != 0:
            raise _friendly_error(result.stderr or result.stdout)
        wait_for_wave(output_path, timeout=output_timeout, probe=probe)
    finally:
        _SYNTHESIS_LOCK.release()


def open_speechnote(*, popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen) -> None:
    try:
        popen(
            ["flatpak", "run", APP_ID],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise SpeechNoteError("Flatpak no está instalado") from exc
