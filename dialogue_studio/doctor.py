"""Command-line environment diagnostic used by ``make doctor``."""

from __future__ import annotations

from .audio import has_ffmpeg
from .paths import resolve_music_dir
from .speechnote import (
    SpeechNoteError,
    check_flatpak,
    check_speechnote_installed,
    check_speechnote_open,
    get_active_tts_model,
    list_tts_models,
)


def main() -> int:
    print("SpeechNote Dialogue Studio · diagnóstico")
    print(f"Flatpak: {'disponible' if check_flatpak() else 'no disponible'}")
    installed = check_speechnote_installed()
    print(f"Speech Note: {'instalado' if installed else 'no instalado'}")
    print(f"Speech Note abierto: {'sí' if check_speechnote_open() else 'no'}")
    print(f"FFmpeg + ffprobe: {'disponibles' if has_ffmpeg() else 'incompletos'}")
    try:
        print(f"Carpeta Música: {resolve_music_dir()}")
        models = list_tts_models() if installed else []
        print(f"Voces TTS: {len(models)}")
        active = get_active_tts_model() if installed else None
        print(f"Voz activa: {active.model_id if active else 'ninguna'}")
    except (RuntimeError, SpeechNoteError) as exc:
        print(f"Aviso: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
