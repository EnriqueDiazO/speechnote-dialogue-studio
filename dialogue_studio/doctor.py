"""Command-line environment diagnostic used by ``make doctor``."""

from __future__ import annotations

from .audio import has_ffmpeg
from .paths import AppPaths, resolve_music_dir
from .recovery import inspect_interrupted_synthesis
from .speechnote import (
    SpeechNoteError,
    check_flatpak,
    check_speechnote_installed,
    check_speechnote_open,
    get_active_tts_model,
    list_tts_models,
)
from .storage import ProjectStore


def main() -> int:
    print("SpeechNote Dialogue Studio · diagnóstico")
    print(f"Flatpak: {'disponible' if check_flatpak() else 'no disponible'}")
    installed = check_speechnote_installed()
    print(f"Speech Note: {'instalado' if installed else 'no instalado'}")
    try:
        is_open = installed and check_speechnote_open()
    except SpeechNoteError:
        is_open = False
    print(f"Speech Note abierto: {'sí' if is_open else 'no'}")
    print(f"FFmpeg + ffprobe: {'disponibles' if has_ffmpeg() else 'incompletos'}")
    try:
        music_dir = resolve_music_dir()
        print(f"Carpeta Música: {music_dir}")
    except RuntimeError as exc:
        print(f"Aviso: {exc}")
        return 1

    if is_open:
        try:
            models = list_tts_models()
            print(f"Voces TTS: {len(models)}")
            active = get_active_tts_model()
            print(f"Voz activa: {active.model_id if active else 'ninguna'}")
        except SpeechNoteError as exc:
            print(f"Aviso TTS: {exc}")
    else:
        print("Voces TTS: diagnóstico omitido (Speech Note está cerrado)")

    paths = AppPaths(music_dir)
    store = ProjectStore(paths)
    generating = 0
    valid = 0
    partial = 0
    missing = 0
    affected_projects = 0
    for record in store.list_projects():
        project = store.load(record.directory)
        generating += sum(utterance.status == "generating" for utterance in project.utterances)
        report = inspect_interrupted_synthesis(project, record.directory)
        if report.items:
            affected_projects += 1
        valid += sum(item.audio_state == "valid" for item in report.items)
        partial += sum(item.audio_state in {"invalid", "mismatched"} for item in report.items)
        missing += sum(item.audio_state == "missing" for item in report.items)
    persistent_locks = list(paths.root.rglob("*.lock")) + list(paths.root.rglob("*.tmp"))
    print(f"Intervenciones generating: {generating}")
    print(f"WAV recuperables válidos: {valid}")
    print(f"WAV parciales o inválidos: {partial}")
    print(f"WAV ausentes en recuperación: {missing}")
    print(f"Locks/temporales persistentes: {len(persistent_locks)}")
    recommendation = (
        "Abrir el proyecto y usar ‘Recuperar síntesis interrumpida’"
        if affected_projects
        else "No se requiere recuperación"
    )
    print(f"Acción recomendada: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
