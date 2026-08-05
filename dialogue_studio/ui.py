"""Streamlit interface for the local dialogue studio."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import streamlit as st

from .audio import export_mp3, has_ffmpeg, probe_audio
from .export import export_project_zip
from .models import (
    DialogueProject,
    SpeakerProfile,
    SpeakerTTSConfig,
    Utterance,
    UtteranceTTSOverride,
    effective_tts_config,
)
from .paths import AppPaths
from .qwen_client import QwenBackendManager, QwenClient, QwenClientError
from .qwen_service import (
    DEFAULT_GENERATION_OPTIONS,
    DEFAULT_MODEL,
    FALLBACK_LANGUAGES,
    FALLBACK_SPEAKERS,
)
from .recovery import (
    RECOVERABLE_MESSAGE,
    inspect_interrupted_synthesis,
    recover_interrupted_synthesis,
)
from .service import (
    add_speaker,
    add_utterance,
    build_master,
    delete_utterance,
    duplicate_utterance,
    generate_utterance,
    move_utterance,
    prepare_generation_paths,
    project_metrics,
    remove_speaker,
    update_speaker_name,
    update_speaker_tts,
    update_speaker_voice,
    update_utterance,
    update_utterance_tts_override,
)
from .speechnote import (
    SpeechNoteError,
    check_flatpak,
    check_speechnote_installed,
    check_speechnote_open,
    get_active_tts_model,
    list_tts_models,
    open_speechnote,
    synthesize_text,
)
from .storage import ProjectStore
from .synthesis import (
    GLOBAL_SYNTHESIS_COORDINATOR,
    SynthesisBusyError,
    run_with_synthesis_state,
)

COLORS = {
    "professor": "#D1A65A",
    "student": "#C68EB5",
    "accent": "#66B3A7",
    "blue": "#78A7C1",
    "sage": "#8FB58A",
    "clay": "#C58E72",
}
STATUS_LABELS = {
    "draft": "Borrador",
    "generating": "Sintetizando",
    "ready": "Listo",
    "error": "Error",
    "stale": "Desactualizado",
}


@dataclass(frozen=True)
class ProjectCapabilities:
    """Independent UI capabilities; TTS health never controls project editing."""

    can_edit_project: bool
    can_synthesize: bool
    can_build_master: bool
    can_export_project: bool
    has_active_synthesis: bool
    active_utterance_id: str | None
    speechnote_available: bool
    qwen_available: bool


@dataclass(frozen=True)
class UtteranceCapabilities:
    can_edit: bool
    can_synthesize: bool
    can_play: bool


CSS = """
<style>
:root { color-scheme: dark; }
.stApp { background: #10171B; }
[data-testid="stSidebar"] { border-right: 1px solid #31434A; }
.block-container { max-width: 1160px; padding-top: 2.2rem; padding-bottom: 5rem; }
h1, h2, h3 { letter-spacing: -0.025em; }
h1 { font-weight: 620; }
.studio-kicker {
    color: #66B3A7; font-size: .76rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; margin-bottom: .35rem;
}
.studio-lead { color: #A8B2B3; max-width: 760px; margin-bottom: 1.4rem; }
.speaker-rule { height: 3px; border-radius: 4px; margin: .15rem 0 .55rem; }
.utterance-head { display:flex; justify-content:space-between; gap:1rem; align-items:center; }
.utterance-number {
    color:#A8B2B3; font-size:.78rem; letter-spacing:.09em; text-transform:uppercase;
}
.status-chip {
    display:inline-block; padding:.14rem .5rem; border:1px solid #49616A; border-radius:999px;
    color:#CFD7D5; font-size:.72rem; letter-spacing:.035em;
}
.system-line { color:#A8B2B3; font-size:.84rem; margin:.15rem 0; }
.system-good { color:#70B98A; }
.system-warn { color:#D4B869; }
.system-bad { color:#D77A72; }
.quiet-note { color:#A8B2B3; font-size:.82rem; }
[data-testid="stMetric"] {
    background:#172227; border:1px solid #31434A; border-radius:.65rem; padding:.7rem .85rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#31434A; }
.stButton button, .stDownloadButton button { border-radius:.45rem; }
</style>
"""


@st.cache_data(ttl=20, show_spinner=False)
def system_diagnostics() -> dict[str, object]:
    data: dict[str, object] = {
        "flatpak": False,
        "installed": False,
        "open": False,
        "ffmpeg": has_ffmpeg(),
        "models": [],
        "active": None,
        "error": None,
        "qwen": {
            "ok": False,
            "state": "offline",
            "model": DEFAULT_MODEL,
            "last_error": None,
        },
        "qwen_capabilities": {
            "speakers": list(FALLBACK_SPEAKERS),
            "languages": list(FALLBACK_LANGUAGES),
            "source": "verified_fallback",
            "supports_instruct": False,
            "supports_voice_design": False,
            "supports_voice_cloning": False,
            "supports_sampling_controls": True,
            "supports_speaker_selection": True,
            "supports_language_selection": True,
            "defaults": dict(DEFAULT_GENERATION_OPTIONS),
        },
    }
    try:
        data["flatpak"] = check_flatpak()
        data["installed"] = check_speechnote_installed()
        if data["installed"]:
            data["open"] = check_speechnote_open()
            if data["open"]:
                data["models"] = list_tts_models()
                data["active"] = get_active_tts_model()
    except (RuntimeError, SpeechNoteError) as exc:
        data["error"] = str(exc)
    qwen = QwenClient()
    try:
        data["qwen"] = qwen.health()
        data["qwen_capabilities"] = qwen.capabilities()
    except (QwenClientError, ValueError):
        pass
    return data


def _init_state() -> tuple[AppPaths, ProjectStore]:
    paths = AppPaths.discover()
    store = ProjectStore(paths)
    if "project" not in st.session_state:
        st.session_state.project = DialogueProject.new()
        st.session_state.project_dir = None
        st.session_state.master_wav = None
        st.session_state.master_mp3 = None
        st.session_state.project_zip = None
        st.session_state.preview_voice = None
        st.session_state.external_actions_enabled = None
        st.session_state.session_token = uuid4().hex
        st.session_state.recovery_notice = None
    # ``busy`` existed in v0.1 and could survive an interrupted Streamlit run forever.
    # It is not evidence of a real process-local synthesis and is safe to migrate away.
    st.session_state.pop("busy", None)
    st.session_state.pop("active_synthesis_id", None)
    if "session_token" not in st.session_state:
        st.session_state.session_token = uuid4().hex
    GLOBAL_SYNTHESIS_COORDINATOR.clear_abandoned()
    return paths, store


def _reset_artifacts() -> None:
    st.session_state.master_wav = None
    st.session_state.master_mp3 = None
    st.session_state.project_zip = None


def _ensure_saved(store: ProjectStore) -> Path:
    project: DialogueProject = st.session_state.project
    current = st.session_state.project_dir
    directory = store.save(project, Path(current) if current else None)
    st.session_state.project_dir = str(directory)
    return directory


def _safe_audio(project_dir: Path, relative: str | None) -> Path | None:
    if not relative:
        return None
    candidate = (project_dir / relative).resolve()
    root = project_dir.resolve()
    if root not in candidate.parents or candidate.is_symlink() or not candidate.is_file():
        return None
    return candidate


def _valid_ready_audio(project_dir: Path | None, utterance: Utterance) -> Path | None:
    if project_dir is None or utterance.status != "ready":
        return None
    candidate = _safe_audio(project_dir, utterance.audio_relative_path)
    if candidate is None:
        return None
    try:
        with candidate.open("rb") as handle:
            header = handle.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            return None
        probe_audio(candidate)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _utterance_capabilities(
    project: DialogueProject,
    utterance: Utterance,
    project_dir: Path | None,
    capabilities: ProjectCapabilities,
) -> tuple[UtteranceCapabilities, Path | None]:
    audio = _valid_ready_audio(project_dir, utterance)
    tts = effective_tts_config(project, utterance)
    provider_available = (
        capabilities.qwen_available
        if tts.provider == "qwen"
        else capabilities.speechnote_available
    )
    return (
        UtteranceCapabilities(
            can_edit=(
                capabilities.can_edit_project
                and capabilities.active_utterance_id != utterance.utterance_id
            ),
            can_synthesize=(
                capabilities.can_synthesize
                and provider_available
                and bool(utterance.text.strip())
                and bool(tts.voice_id)
            ),
            can_play=audio is not None,
        ),
        audio,
    )


def _project_capabilities(
    project: DialogueProject,
    diagnostics: dict[str, object],
    project_dir: Path | None,
) -> ProjectCapabilities:
    active = GLOBAL_SYNTHESIS_COORDINATOR.active
    has_active_synthesis = active is not None
    external_state = diagnostics.get(
        "external_actions_enabled", st.session_state.external_actions_enabled
    )
    tts_available = (
        bool(diagnostics["installed"]) and bool(diagnostics["open"]) and external_state is not False
    )
    qwen_status = diagnostics.get("qwen", {})
    qwen_state = qwen_status.get("state", "offline") if isinstance(qwen_status, dict) else "offline"
    qwen_available = qwen_state in {"idle", "error"}
    playable = [
        _valid_ready_audio(project_dir, utterance) is not None for utterance in project.utterances
    ]
    return ProjectCapabilities(
        can_edit_project=True,
        can_synthesize=(tts_available or qwen_available) and not has_active_synthesis,
        can_build_master=(bool(playable) and all(playable) and not has_active_synthesis),
        can_export_project=bool(project.utterances),
        has_active_synthesis=has_active_synthesis,
        active_utterance_id=active.utterance_id if active else None,
        speechnote_available=tts_available,
        qwen_available=qwen_available,
    )


def _system_line(label: str, state: bool | None, detail: str = "") -> None:
    if state is True:
        marker, css = "●", "system-good"
    elif state is False:
        marker, css = "●", "system-bad"
    else:
        marker, css = "◆", "system-warn"
    st.markdown(
        f'<div class="system-line"><span class="{css}">{marker}</span> '
        f"{html.escape(label)} {html.escape(detail)}</div>",
        unsafe_allow_html=True,
    )


def _render_sidebar(
    paths: AppPaths,
    store: ProjectStore,
    diagnostics: dict[str, object],
    capabilities: ProjectCapabilities,
) -> None:
    project: DialogueProject = st.session_state.project
    with st.sidebar:
        st.markdown("### SpeechNote Dialogue Studio")
        st.caption("Estudio local de diálogo")
        st.markdown("#### Estado del sistema")
        _system_line("Speech Note", bool(diagnostics["installed"]))
        _system_line("Aplicación abierta", bool(diagnostics["open"]))
        external_state = diagnostics.get(
            "external_actions_enabled", st.session_state.external_actions_enabled
        )
        if not diagnostics["open"]:
            external_state = False
        _system_line(
            "Invocación externa",
            external_state,
            "por comprobar"
            if diagnostics["open"] and external_state is None
            else "requiere Speech Note abierto"
            if not diagnostics["open"]
            else "",
        )
        _system_line("FFmpeg", bool(diagnostics["ffmpeg"]))
        st.caption(f"Música · {paths.music_dir}")
        if diagnostics["error"]:
            st.warning(str(diagnostics["error"]))
        columns = st.columns(2)
        if columns[0].button("Actualizar", use_container_width=True):
            system_diagnostics.clear()
            st.rerun()
        if columns[1].button("Abrir app", use_container_width=True):
            try:
                open_speechnote()
                st.toast("Speech Note se está abriendo")
            except SpeechNoteError as exc:
                st.error(str(exc))

        st.divider()
        st.markdown("#### Proyecto")
        actions = st.columns(2)
        if actions[0].button(
            "Nuevo",
            use_container_width=True,
            disabled=not capabilities.can_edit_project,
        ):
            st.session_state.project = DialogueProject.new()
            st.session_state.project_dir = None
            _reset_artifacts()
            st.rerun()
        if actions[1].button(
            "Ejemplo",
            use_container_width=True,
            disabled=not capabilities.can_edit_project,
        ):
            st.session_state.project = DialogueProject.sample()
            st.session_state.project_dir = None
            _reset_artifacts()
            st.rerun()

        records = store.list_projects()
        if records:
            labels = {
                record.project_id: f"{record.title} · {record.project_id[:8]}"
                for record in records
            }
            selected = st.selectbox(
                "Proyectos guardados",
                options=[record.project_id for record in records],
                format_func=lambda value: labels[value],
                key="open_project_choice",
            )
            if st.button(
                "Abrir seleccionado",
                use_container_width=True,
                disabled=not capabilities.can_edit_project,
            ):
                record = next(item for item in records if item.project_id == selected)
                st.session_state.project = store.load(record.directory)
                st.session_state.project_dir = str(record.directory)
                _reset_artifacts()
                st.rerun()
        else:
            st.caption("Aún no hay proyectos guardados.")

        title = st.text_input(
            "Título",
            project.title,
            key=f"title-{project.project_id}",
            disabled=not capabilities.can_edit_project,
        )
        if title != project.title:
            project.title = title.strip() or "Diálogo sin título"
            project.touch()
        if st.button(
            "Guardar proyecto",
            type="primary",
            use_container_width=True,
            disabled=not capabilities.can_edit_project,
        ):
            try:
                _ensure_saved(store)
                st.toast("Proyecto guardado")
            except (OSError, ValueError) as exc:
                st.error(str(exc))

        st.divider()
        st.markdown("#### Configuración")
        pause = st.number_input(
            "Pausa entre intervenciones (ms)",
            min_value=0,
            max_value=5000,
            value=project.pause_ms,
            step=50,
            key=f"pause-{project.project_id}",
        )
        if pause != project.pause_ms:
            project.pause_ms = int(pause)
            project.touch()
            _reset_artifacts()
        st.checkbox("Incluir WAV", value=True, disabled=True)
        st.checkbox(
            "Habilitar MP3",
            value=bool(diagnostics["ffmpeg"]),
            disabled=not bool(diagnostics["ffmpeg"]),
            key="enable_mp3",
        )


def _render_header(project: DialogueProject, capabilities: ProjectCapabilities) -> None:
    st.markdown('<div class="studio-kicker">Estudio de voz académico</div>', unsafe_allow_html=True)
    st.title(project.title)
    st.markdown(
        '<p class="studio-lead">Escribe, asigna voces y ensambla un diálogo local sin '
        "enviar el texto ni el audio a la nube.</p>",
        unsafe_allow_html=True,
    )
    description = st.text_area(
        "Descripción",
        project.description,
        height=72,
        placeholder="Contexto, objetivo o notas del diálogo…",
        key=f"description-{project.project_id}",
        disabled=not capabilities.can_edit_project,
    )
    if description != project.description:
        project.description = description
        project.touch()
    metrics = project_metrics(project)
    columns = st.columns(4)
    columns[0].metric("Intervenciones", metrics["utterances"])
    columns[1].metric("Generadas", metrics["generated"])
    columns[2].metric("Duración", f"{metrics['duration_seconds']:.1f} s")
    columns[3].metric("Pendientes", metrics["pending"])


def _clear_diagnostics_cache() -> None:
    clear = getattr(system_diagnostics, "clear", None)
    if callable(clear):
        clear()


def _gib(value: object) -> str:
    return f"{int(value) / (1024**3):.1f} GiB" if isinstance(value, int) else "—"


def _render_qwen_backend(paths: AppPaths, diagnostics: dict[str, object]) -> None:
    raw_status = diagnostics.get("qwen", {})
    status = raw_status if isinstance(raw_status, dict) else {}
    raw_capabilities = diagnostics.get("qwen_capabilities", {})
    qwen_capabilities = raw_capabilities if isinstance(raw_capabilities, dict) else {}
    state = str(status.get("state", "offline"))
    state_labels = {
        "offline": "Apagado",
        "starting": "Iniciando",
        "idle": "Disponible",
        "loading_model": "Cargando modelo",
        "generating": "Generando",
        "error": "Error",
    }
    with st.expander("Backend Qwen3-TTS", expanded=state != "offline"):
        summary = st.columns(4)
        summary[0].metric("Estado", state_labels.get(state, state))
        summary[1].metric("GPU", str(status.get("gpu") or "—"))
        summary[2].metric("VRAM libre", _gib(status.get("vram_free_bytes")))
        summary[3].metric("VRAM total", _gib(status.get("vram_total_bytes")))
        st.caption(
            f"Modelo · {status.get('model', DEFAULT_MODEL)}  |  "
            f"Torch · {status.get('torch', '—')}  |  CUDA · {status.get('cuda_version', '—')}  |  "
            f"BF16 · {'sí' if status.get('bf16') else '—'}  |  "
            f"dtype · {status.get('dtype', 'bfloat16')}  |  "
            f"atención · {status.get('attention', 'sdpa')}  |  salida · 24000 Hz"
        )
        speakers = qwen_capabilities.get("speakers", FALLBACK_SPEAKERS)
        languages = qwen_capabilities.get("languages", FALLBACK_LANGUAGES)
        st.caption(f"Voces · {len(speakers)}  |  Idiomas · {len(languages)}")
        st.info(
            "El modelo 0.6B permite elegir voz, idioma y sampling. No admite instrucciones "
            "emocionales, VoiceDesign ni clonación de voz; esos controles permanecen ocultos."
        )
        if status.get("last_error"):
            st.error(str(status["last_error"]))
        controls = st.columns(4)
        manager = QwenBackendManager(paths)
        if controls[0].button(
            "Iniciar backend Qwen",
            disabled=state != "offline",
            use_container_width=True,
        ):
            try:
                manager.start()
                _clear_diagnostics_cache()
                st.rerun()
            except (OSError, ValueError, QwenClientError) as exc:
                st.error(str(exc))
        if controls[1].button("Actualizar estado", use_container_width=True):
            _clear_diagnostics_cache()
            st.rerun()
        if controls[2].button(
            "Descargar modelo de GPU",
            disabled=state == "offline" or not bool(status.get("model_loaded")),
            use_container_width=True,
        ):
            try:
                manager.client.unload()
                _clear_diagnostics_cache()
                st.rerun()
            except QwenClientError as exc:
                st.error(str(exc))
        if controls[3].button(
            "Detener backend",
            disabled=state == "offline",
            use_container_width=True,
        ):
            try:
                manager.stop()
                _clear_diagnostics_cache()
                st.rerun()
            except (OSError, QwenClientError) as exc:
                st.error(str(exc))


def _render_recovery(
    project: DialogueProject,
    store: ProjectStore,
    capabilities: ProjectCapabilities,
) -> None:
    notice = st.session_state.get("recovery_notice")
    if notice:
        st.success(str(notice))
        st.session_state.recovery_notice = None
    current = st.session_state.project_dir
    if not current:
        return
    directory = Path(current)
    report = inspect_interrupted_synthesis(project, directory)
    if not report.items:
        return
    st.warning(
        f"Se detectaron {report.affected_count} síntesis interrumpidas. "
        "El proyecto sigue siendo editable."
    )
    proposed_labels = {
        "mark_ready": "Recuperar WAV válido como listo",
        "mark_stale": "Marcar como pendiente de regeneración",
        "preserve_partial": "Conservar como .partial y marcar pendiente",
    }
    audio_labels = {
        "valid": "válido",
        "missing": "ausente",
        "invalid": "inválido",
        "mismatched": "no corresponde a la intervención",
    }
    with st.expander("Detalle de recuperación"):
        for item in report.items:
            st.write(
                f"Intervención {item.order:02d} · WAV {audio_labels[item.audio_state]} · "
                f"{proposed_labels[item.proposed_action]}"
            )
    if st.button(
        "Recuperar síntesis interrumpida",
        type="primary",
        disabled=capabilities.has_active_synthesis,
    ):
        recovered = recover_interrupted_synthesis(project, directory)
        if recovered.changed:
            store.save(project, directory)
            st.session_state.recovery_notice = (
                f"Recuperación completada: {recovered.recovered_ready} WAV válidos y "
                f"{recovered.converted_recoverable} pendientes."
            )
        st.rerun()


def _model_options(
    project: DialogueProject, diagnostics: dict[str, object]
) -> tuple[list[str], dict[str, str]]:
    models = list(diagnostics["models"])
    labels = {model.model_id: model.label for model in models}
    for speaker in project.speakers:
        config = speaker.tts_config
        if (
            config.provider == "speechnote"
            and config.voice_id
            and config.voice_id not in labels
        ):
            labels[config.voice_id] = config.voice_label or config.voice_id
    return list(labels), labels


QWEN_VOICE_LABELS = {
    "aiden": "Aiden",
    "dylan": "Dylan",
    "eric": "Eric",
    "ono_anna": "Ono Anna",
    "ryan": "Ryan",
    "serena": "Serena",
    "sohee": "Sohee",
    "uncle_fu": "Uncle Fu",
    "vivian": "Vivian",
}
QWEN_LANGUAGE_LABELS = {
    "auto": "Automático",
    "chinese": "Chino",
    "english": "Inglés",
    "french": "Francés",
    "german": "Alemán",
    "italian": "Italiano",
    "japanese": "Japonés",
    "korean": "Coreano",
    "portuguese": "Portugués",
    "russian": "Ruso",
    "spanish": "Español",
}
PROVIDER_LABELS = {"speechnote": "Speech Note", "qwen": "Qwen3-TTS"}


def _qwen_options(diagnostics: dict[str, object]) -> tuple[list[str], list[str]]:
    raw = diagnostics.get("qwen_capabilities", {})
    capabilities = raw if isinstance(raw, dict) else {}
    speakers = [str(value).lower() for value in capabilities.get("speakers", FALLBACK_SPEAKERS)]
    languages = [str(value).lower() for value in capabilities.get("languages", FALLBACK_LANGUAGES)]
    return speakers or list(FALLBACK_SPEAKERS), languages or list(FALLBACK_LANGUAGES)


def _qwen_defaults(config: SpeakerTTSConfig) -> dict[str, int | float]:
    options = dict(DEFAULT_GENERATION_OPTIONS)
    options.update(config.generation_options)
    return options


def _sampling_controls(
    config: SpeakerTTSConfig,
    *,
    key_prefix: str,
    disabled: bool,
) -> tuple[dict[str, int | float], bool]:
    reset_marker = f"{key_prefix}-reset-pending"
    suffixes = ("seed", "max-new-tokens", "temperature", "top-p", "top-k", "repetition")
    if st.session_state.pop(reset_marker, False):
        for suffix in suffixes:
            st.session_state.pop(f"{key_prefix}-{suffix}", None)
    values = _qwen_defaults(config)
    seed = st.number_input(
        "Semilla",
        min_value=0,
        max_value=4_294_967_295,
        value=int(values["seed"]),
        step=1,
        key=f"{key_prefix}-seed",
        disabled=disabled,
        help="Se aplica dentro de un contexto Torch que restaura el RNG al terminar.",
    )
    max_tokens = st.number_input(
        "Máximo de tokens",
        min_value=64,
        max_value=8192,
        value=int(values["max_new_tokens"]),
        step=64,
        key=f"{key_prefix}-max-new-tokens",
        disabled=disabled,
    )
    columns = st.columns(2)
    temperature = columns[0].number_input(
        "Temperatura",
        min_value=0.1,
        max_value=2.0,
        value=float(values["temperature"]),
        step=0.05,
        key=f"{key_prefix}-temperature",
        disabled=disabled,
    )
    top_p = columns[1].number_input(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=float(values["top_p"]),
        step=0.05,
        key=f"{key_prefix}-top-p",
        disabled=disabled,
    )
    top_k = columns[0].number_input(
        "Top-k",
        min_value=1,
        max_value=200,
        value=int(values["top_k"]),
        step=1,
        key=f"{key_prefix}-top-k",
        disabled=disabled,
    )
    repetition = columns[1].number_input(
        "Penalización de repetición",
        min_value=0.8,
        max_value=2.0,
        value=float(values["repetition_penalty"]),
        step=0.05,
        key=f"{key_prefix}-repetition",
        disabled=disabled,
    )
    reset = st.button(
        "Restablecer opciones",
        key=f"{key_prefix}-reset",
        disabled=disabled,
    )
    if reset:
        st.session_state[reset_marker] = True
    return (
        dict(DEFAULT_GENERATION_OPTIONS)
        if reset
        else {
            "seed": int(seed),
            "max_new_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition),
        },
        reset,
    )


def _test_voice(paths: AppPaths, speaker: SpeakerProfile) -> None:
    config = speaker.tts_config
    destination = paths.temporary / f"voice-test-{uuid4().hex}.wav"
    with st.spinner(f"Probando la voz de {speaker.name}…"):
        if config.provider == "qwen":
            def action() -> object:
                return QwenClient().synthesize(
                    text=f"Hola, soy {speaker.name}. Esta es una prueba de voz.",
                    speaker=config.voice_id,
                    language=config.language,
                    generation_options=_qwen_defaults(config),
                    output_path=destination,
                )
        else:
            def action() -> object:
                return synthesize_text(
                    config.voice_id,
                    f"Hola, soy {speaker.name}. Esta es una prueba de voz.",
                    destination,
                    paths.root,
                    probe=probe_audio,
                )
        run_with_synthesis_state(
            GLOBAL_SYNTHESIS_COORDINATOR,
            f"voice-test-{config.provider}-{config.voice_id}",
            destination,
            st.session_state.session_token,
            action,
        )
        probe_audio(destination)
    previous = st.session_state.preview_voice
    if previous:
        previous_path = Path(previous)
        if (
            previous_path.parent == paths.temporary
            and previous_path.name.startswith("voice-test-")
            and previous_path.suffix == ".wav"
        ):
            previous_path.unlink(missing_ok=True)
    st.session_state.preview_voice = str(destination)
    st.toast("Prueba de voz lista")


def _render_speakers(
    project: DialogueProject,
    paths: AppPaths,
    diagnostics: dict[str, object],
    capabilities: ProjectCapabilities,
) -> None:
    st.markdown("## Hablantes")
    st.caption("Cada voz se aplica a todas las intervenciones de ese hablante.")
    speech_options, speech_labels = _model_options(project, diagnostics)
    qwen_voices, qwen_languages = _qwen_options(diagnostics)
    for speaker in list(project.speakers):
        config = speaker.tts_config
        color = COLORS.get(speaker.color_key, COLORS["accent"])
        with st.container(border=True, key=f"speaker-card-{speaker.speaker_id}"):
            st.markdown(
                f'<div class="speaker-rule" style="background:{color}"></div>',
                unsafe_allow_html=True,
            )
            columns = st.columns([1.1, 1.1, 0.8])
            name = columns[0].text_input(
                "Nombre",
                speaker.name,
                key=f"speaker-name-{speaker.speaker_id}",
                disabled=not capabilities.can_edit_project,
            )
            if name.strip() and name != speaker.name:
                update_speaker_name(project, speaker.speaker_id, name)
                _reset_artifacts()
            provider = columns[1].selectbox(
                "Motor de voz",
                list(PROVIDER_LABELS),
                index=list(PROVIDER_LABELS).index(config.provider),
                format_func=lambda value: PROVIDER_LABELS[value],
                key=f"speaker-provider-{speaker.speaker_id}",
                disabled=not capabilities.can_edit_project,
            )
            if provider != config.provider:
                if provider == "qwen":
                    replacement = SpeakerTTSConfig(
                        provider="qwen",
                        voice_id=qwen_voices[0],
                        voice_label=QWEN_VOICE_LABELS.get(qwen_voices[0], qwen_voices[0]),
                        language="spanish" if project.language.startswith("es") else "auto",
                        generation_options=dict(DEFAULT_GENERATION_OPTIONS),
                    )
                else:
                    voice_id = speech_options[0] if speech_options else ""
                    replacement = SpeakerTTSConfig(
                        provider="speechnote",
                        voice_id=voice_id,
                        voice_label=speech_labels.get(voice_id, voice_id),
                    )
                update_speaker_tts(project, speaker.speaker_id, replacement)
                _reset_artifacts()
                st.rerun()
            color_key = columns[2].selectbox(
                "Color",
                list(COLORS),
                index=list(COLORS).index(speaker.color_key) if speaker.color_key in COLORS else 2,
                format_func=lambda value: value.capitalize(),
                key=f"speaker-color-{speaker.speaker_id}",
                disabled=not capabilities.can_edit_project,
            )
            if color_key != speaker.color_key:
                speaker.color_key = color_key
                project.touch()

            if config.provider == "qwen":
                voice_id = config.voice_id if config.voice_id in qwen_voices else qwen_voices[0]
                voice = st.selectbox(
                    "Voz",
                    qwen_voices,
                    index=qwen_voices.index(voice_id),
                    format_func=lambda value: QWEN_VOICE_LABELS.get(value, value),
                    key=f"speaker-voice-{speaker.speaker_id}",
                    disabled=not capabilities.can_edit_project,
                )
                language_id = (
                    config.language if config.language in qwen_languages else qwen_languages[0]
                )
                language = st.selectbox(
                    "Idioma",
                    qwen_languages,
                    index=qwen_languages.index(language_id),
                    format_func=lambda value: QWEN_LANGUAGE_LABELS.get(value, value.title()),
                    key=f"speaker-language-{speaker.speaker_id}",
                    disabled=not capabilities.can_edit_project,
                )
                st.caption(
                    "Qwen 0.6B no admite instrucciones emocionales; sólo se envían voz, "
                    "idioma y opciones de sampling confirmadas."
                )
                with st.expander("Opciones avanzadas de generación"):
                    generation_options, reset = _sampling_controls(
                        config,
                        key_prefix=f"speaker-qwen-{speaker.speaker_id}",
                        disabled=not capabilities.can_edit_project,
                    )
                updated = SpeakerTTSConfig(
                    provider="qwen",
                    voice_id=voice,
                    voice_label=QWEN_VOICE_LABELS.get(voice, voice),
                    language=language,
                    generation_options=generation_options,
                )
                if updated != config:
                    update_speaker_tts(project, speaker.speaker_id, updated)
                    _reset_artifacts()
                    if reset:
                        st.rerun()
            else:
                if speech_options:
                    current = (
                        config.voice_id if config.voice_id in speech_options else speech_options[0]
                    )
                    voice = st.selectbox(
                        "Voz",
                        speech_options,
                        index=speech_options.index(current),
                        format_func=lambda value: speech_labels[value],
                        key=f"speaker-voice-{speaker.speaker_id}",
                        disabled=not capabilities.can_edit_project,
                    )
                    if voice != config.voice_id:
                        update_speaker_voice(
                            project, speaker.speaker_id, voice, speech_labels[voice]
                        )
                        _reset_artifacts()
                else:
                    st.text_input(
                        "ID de voz",
                        config.voice_id,
                        disabled=True,
                        help="Abre Speech Note y actualiza el diagnóstico.",
                    )
            actions = st.columns([1, 1, 3])
            provider_available = (
                capabilities.qwen_available
                if config.provider == "qwen"
                else capabilities.speechnote_available
            )
            if actions[0].button(
                "Probar voz",
                key=f"test-voice-{speaker.speaker_id}",
                disabled=(
                    not config.voice_id
                    or not provider_available
                    or capabilities.has_active_synthesis
                ),
            ):
                try:
                    _test_voice(paths, speaker)
                    _clear_diagnostics_cache()
                except (
                    OSError,
                    ValueError,
                    SpeechNoteError,
                    QwenClientError,
                    SynthesisBusyError,
                ) as exc:
                    st.error(str(exc))
            in_use = any(item.speaker_id == speaker.speaker_id for item in project.utterances)
            if actions[1].button(
                "Eliminar",
                key=f"delete-speaker-{speaker.speaker_id}",
                disabled=in_use or len(project.speakers) == 1,
                help="Reasigna sus intervenciones primero." if in_use else None,
            ):
                try:
                    remove_speaker(project, speaker.speaker_id)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    preview = st.session_state.preview_voice
    if preview and Path(preview).is_file():
        st.caption("Última prueba de voz")
        st.audio(preview, format="audio/wav")

    with st.expander("Añadir hablante"):
        name = st.text_input(
            "Nombre",
            placeholder="Narradora",
            key="new-speaker-name",
            disabled=not capabilities.can_edit_project,
        )
        new_provider = st.selectbox(
            "Motor de voz",
            list(PROVIDER_LABELS),
            format_func=lambda value: PROVIDER_LABELS[value],
            key="new-speaker-provider",
            disabled=not capabilities.can_edit_project,
        )
        if new_provider == "qwen":
            new_voice = st.selectbox(
                "Voz Qwen",
                qwen_voices,
                format_func=lambda value: QWEN_VOICE_LABELS.get(value, value),
                key="new-speaker-qwen-voice",
                disabled=not capabilities.can_edit_project,
            )
            new_language = st.selectbox(
                "Idioma Qwen",
                qwen_languages,
                index=(
                    qwen_languages.index("spanish") if "spanish" in qwen_languages else 0
                ),
                format_func=lambda value: QWEN_LANGUAGE_LABELS.get(value, value.title()),
                key="new-speaker-qwen-language",
                disabled=not capabilities.can_edit_project,
            )
            new_tts = SpeakerTTSConfig(
                provider="qwen",
                voice_id=new_voice,
                voice_label=QWEN_VOICE_LABELS.get(new_voice, new_voice),
                language=new_language,
                generation_options=dict(DEFAULT_GENERATION_OPTIONS),
            )
        else:
            new_voice = st.selectbox(
                "Voz Speech Note",
                speech_options or [""],
                format_func=lambda value: speech_labels.get(value, "Sin voces detectadas"),
                key="new-speaker-speechnote-voice",
                disabled=not capabilities.can_edit_project,
            )
            new_tts = SpeakerTTSConfig(
                provider="speechnote",
                voice_id=new_voice,
                voice_label=speech_labels.get(new_voice, new_voice),
            )
        color = st.selectbox(
            "Color",
            list(COLORS),
            index=2,
            format_func=lambda value: value.capitalize(),
            disabled=not capabilities.can_edit_project,
        )
        if st.button(
            "Crear hablante",
            type="primary",
            key="create-speaker",
            disabled=not capabilities.can_edit_project,
        ):
            if not name.strip():
                st.error("Escribe un nombre para el hablante")
            else:
                add_speaker(
                    project,
                    name,
                    new_tts.voice_id,
                    new_tts.voice_label,
                    color,
                    tts=new_tts,
                )
                st.rerun()


def _run_generation(
    project: DialogueProject,
    store: ProjectStore,
    paths: AppPaths,
    utterance_ids: list[str],
) -> None:
    active = GLOBAL_SYNTHESIS_COORDINATOR.active
    if active is not None:
        active_utterance = next(
            (item for item in project.utterances if item.utterance_id == active.utterance_id),
            None,
        )
        label = f" {active_utterance.order:02d}" if active_utterance else ""
        st.info(f"Sintetizando intervención{label}…")
        return
    directory = _ensure_saved(store)
    progress = st.progress(0, text="Preparando síntesis…")
    failures = 0
    try:
        total = len(utterance_ids)
        for index, utterance_id in enumerate(utterance_ids, start=1):
            utterance = next(
                item for item in project.utterances if item.utterance_id == utterance_id
            )
            output_paths = prepare_generation_paths(directory, utterance, paths.root)
            voice = project.speaker(utterance.speaker_id)
            progress.progress(
                (index - 1) / total,
                text=f"Intervención {index} de {total} · {voice.name} · sintetizando",
            )
            try:
                run_with_synthesis_state(
                    GLOBAL_SYNTHESIS_COORDINATOR,
                    utterance_id,
                    output_paths.raw,
                    st.session_state.session_token,
                    lambda current_id=utterance_id, current_paths=output_paths: generate_utterance(
                        project,
                        directory,
                        current_id,
                        paths.root,
                        output_paths=current_paths,
                    ),
                )
                st.session_state.external_actions_enabled = True
            except SynthesisBusyError:
                failures += 1
                st.info(f"Sintetizando intervención {utterance.order:02d}…")
            except (OSError, RuntimeError, ValueError) as exc:
                failures += 1
                if "invocación externa está deshabilitada" in str(exc).lower():
                    st.session_state.external_actions_enabled = False
                st.error(f"Intervención {utterance.order}: {exc}")
            finally:
                store.save(project, directory)
            progress.progress(index / total, text=f"Intervención {index} de {total} completada")
        _reset_artifacts()
        if not failures:
            st.toast("Síntesis completada")
    finally:
        GLOBAL_SYNTHESIS_COORDINATOR.clear_abandoned()


def _generate_one(
    project: DialogueProject,
    store: ProjectStore,
    paths: AppPaths,
    utterance_id: str,
) -> None:
    _run_generation(project, store, paths, [utterance_id])


def _render_utterance_tts_override(
    project: DialogueProject,
    utterance: Utterance,
    diagnostics: dict[str, object],
    *,
    disabled: bool,
) -> None:
    speech_options, speech_labels = _model_options(project, diagnostics)
    qwen_voices, qwen_languages = _qwen_options(diagnostics)
    existing = utterance.tts_override
    enabled = st.checkbox(
        "Usar configuración de voz específica",
        value=existing is not None,
        key=f"utterance-override-enabled-{utterance.utterance_id}",
        disabled=disabled,
    )
    if enabled != (existing is not None):
        if enabled:
            base = project.speaker(utterance.speaker_id).tts_config
            replacement = UtteranceTTSOverride(
                provider=base.provider,
                voice_id=base.voice_id,
                language=base.language,
                generation_options=dict(base.generation_options),
            )
        else:
            replacement = None
        update_utterance_tts_override(project, utterance.utterance_id, replacement)
        _reset_artifacts()
        st.rerun()
    if not enabled or existing is None:
        return

    effective = effective_tts_config(project, utterance)
    provider = st.selectbox(
        "Motor de voz para esta intervención",
        list(PROVIDER_LABELS),
        index=list(PROVIDER_LABELS).index(effective.provider),
        format_func=lambda value: PROVIDER_LABELS[value],
        key=f"utterance-provider-{utterance.utterance_id}",
        disabled=disabled,
    )
    if provider == "qwen":
        current_voice = (
            effective.voice_id if effective.voice_id in qwen_voices else qwen_voices[0]
        )
        voice = st.selectbox(
            "Voz Qwen para esta intervención",
            qwen_voices,
            index=qwen_voices.index(current_voice),
            format_func=lambda value: QWEN_VOICE_LABELS.get(value, value),
            key=f"utterance-qwen-voice-{utterance.utterance_id}",
            disabled=disabled,
        )
        current_language = (
            effective.language if effective.language in qwen_languages else qwen_languages[0]
        )
        language = st.selectbox(
            "Idioma Qwen para esta intervención",
            qwen_languages,
            index=qwen_languages.index(current_language),
            format_func=lambda value: QWEN_LANGUAGE_LABELS.get(value, value.title()),
            key=f"utterance-qwen-language-{utterance.utterance_id}",
            disabled=disabled,
        )
        st.caption("El modelo 0.6B no admite un override de instrucción emocional.")
        with st.expander("Opciones avanzadas del override"):
            options, reset = _sampling_controls(
                effective,
                key_prefix=f"utterance-qwen-{utterance.utterance_id}",
                disabled=disabled,
            )
    else:
        current_voice = (
            effective.voice_id if effective.voice_id in speech_options else ""
        )
        choices = speech_options or [current_voice]
        voice = st.selectbox(
            "Voz Speech Note para esta intervención",
            choices,
            index=choices.index(current_voice) if current_voice in choices else 0,
            format_func=lambda value: speech_labels.get(value, value or "Sin voz detectada"),
            key=f"utterance-speechnote-voice-{utterance.utterance_id}",
            disabled=disabled,
        )
        language = "auto"
        options = {}
        reset = False
    updated = UtteranceTTSOverride(
        provider=provider,
        voice_id=voice,
        language=language,
        generation_options=options,
    )
    if updated != existing:
        update_utterance_tts_override(project, utterance.utterance_id, updated)
        _reset_artifacts()
        if reset:
            st.rerun()


def _render_utterances(
    project: DialogueProject,
    store: ProjectStore,
    paths: AppPaths,
    diagnostics: dict[str, object],
    capabilities: ProjectCapabilities,
) -> None:
    st.markdown("## Intervenciones")
    st.caption("La síntesis sólo comienza al pulsar un botón de generación.")
    speaker_ids = [speaker.speaker_id for speaker in project.speakers]
    speaker_names = {speaker.speaker_id: speaker.name for speaker in project.speakers}
    directory = Path(st.session_state.project_dir) if st.session_state.project_dir else None
    for index, utterance in enumerate(list(project.utterances)):
        utterance_capabilities, audio = _utterance_capabilities(
            project, utterance, directory, capabilities
        )
        speaker = project.speaker(utterance.speaker_id)
        color = COLORS.get(speaker.color_key, COLORS["accent"])
        with st.container(border=True, key=f"utterance-card-{utterance.utterance_id}"):
            st.markdown(
                f'<div class="speaker-rule" style="background:{color}"></div>'
                '<div class="utterance-head">'
                f'<span class="utterance-number">Intervención {utterance.order:02d}</span>'
                f'<span class="status-chip">{STATUS_LABELS[utterance.status]}</span></div>',
                unsafe_allow_html=True,
            )
            selected_speaker = st.selectbox(
                "Hablante",
                speaker_ids,
                index=speaker_ids.index(utterance.speaker_id),
                format_func=lambda value: speaker_names[value],
                key=f"utterance-speaker-{utterance.utterance_id}",
                disabled=not utterance_capabilities.can_edit,
            )
            text = st.text_area(
                "Texto",
                utterance.text,
                height=112,
                placeholder="Escribe esta intervención…",
                key=f"utterance-text-{utterance.utterance_id}",
                disabled=not utterance_capabilities.can_edit,
            )
            if selected_speaker != utterance.speaker_id or text != utterance.text:
                update_utterance(
                    project,
                    utterance.utterance_id,
                    text=text,
                    speaker_id=selected_speaker,
                )
                _reset_artifacts()
            with st.expander("Override de voz"):
                _render_utterance_tts_override(
                    project,
                    utterance,
                    diagnostics,
                    disabled=not utterance_capabilities.can_edit,
                )
            if utterance.duration_seconds is not None:
                st.caption(f"Duración · {utterance.duration_seconds:.2f} s")
            if capabilities.active_utterance_id == utterance.utterance_id:
                st.info(f"Sintetizando intervención {utterance.order:02d}…")
            elif utterance.status in {"error", "stale"}:
                st.warning(RECOVERABLE_MESSAGE)
            utterance_capabilities, audio = _utterance_capabilities(
                project, utterance, directory, capabilities
            )
            if utterance_capabilities.can_play and audio:
                with st.expander("Escuchar"):
                    st.audio(str(audio), format="audio/wav")
            actions = st.columns([1.4, 0.75, 0.75, 0.75, 0.85])
            generate_label = (
                "Regenerar"
                if utterance.audio_relative_path or utterance.status in {"ready", "stale", "error"}
                else "Generar"
            )
            if actions[0].button(
                generate_label,
                key=f"generate-{utterance.utterance_id}",
                type="primary",
                disabled=not utterance_capabilities.can_synthesize,
            ):
                _generate_one(project, store, paths, utterance.utterance_id)
                st.rerun()
            if actions[1].button(
                "↑",
                key=f"up-{utterance.utterance_id}",
                disabled=index == 0 or not utterance_capabilities.can_edit,
                help="Subir",
            ):
                move_utterance(project, utterance.utterance_id, -1)
                _reset_artifacts()
                st.rerun()
            if actions[2].button(
                "↓",
                key=f"down-{utterance.utterance_id}",
                disabled=(
                    index == len(project.utterances) - 1 or not utterance_capabilities.can_edit
                ),
                help="Bajar",
            ):
                move_utterance(project, utterance.utterance_id, 1)
                _reset_artifacts()
                st.rerun()
            if actions[3].button(
                "Duplicar",
                key=f"duplicate-{utterance.utterance_id}",
                disabled=not utterance_capabilities.can_edit,
            ):
                duplicate_utterance(project, utterance.utterance_id)
                _reset_artifacts()
                st.rerun()
            if actions[4].button(
                "Eliminar",
                key=f"delete-{utterance.utterance_id}",
                disabled=not utterance_capabilities.can_edit,
            ):
                delete_utterance(project, utterance.utterance_id)
                _reset_artifacts()
                st.rerun()
    if st.button("＋ Añadir intervención", disabled=not capabilities.can_edit_project):
        add_utterance(project)
        st.rerun()


def _render_global_actions(
    project: DialogueProject,
    store: ProjectStore,
    paths: AppPaths,
    diagnostics: dict[str, object],
) -> None:
    project_dir = Path(st.session_state.project_dir) if st.session_state.project_dir else None
    capabilities = _project_capabilities(project, diagnostics, project_dir)
    st.markdown("## Mezcla y exportación")
    pending = [
        item.utterance_id
        for item in project.utterances
        if item.status in {"draft", "stale", "error"}
        and item.text.strip()
        and project.speaker(item.speaker_id).model_id
    ]
    all_nonempty = [
        item.utterance_id
        for item in project.utterances
        if item.text.strip() and project.speaker(item.speaker_id).model_id
    ]
    can_generate_all = len(all_nonempty) == len(project.utterances) and bool(all_nonempty)
    controls = st.columns(4)
    if controls[0].button(
        "Generar pendientes",
        type="primary",
        disabled=not pending or not capabilities.can_synthesize,
        use_container_width=True,
    ):
        _run_generation(project, store, paths, pending)
        st.rerun()
    if controls[1].button(
        "Generar todas",
        disabled=not can_generate_all or not capabilities.can_synthesize,
        use_container_width=True,
    ):
        _run_generation(project, store, paths, all_nonempty)
        st.rerun()
    if controls[2].button(
        "Construir diálogo",
        disabled=not capabilities.can_build_master,
        use_container_width=True,
    ):
        try:
            directory = _ensure_saved(store)
            destination, _ = build_master(project, directory)
            st.session_state.master_wav = str(destination)
            st.session_state.master_mp3 = None
            st.session_state.project_zip = None
            st.toast("WAV maestro listo")
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
    master = Path(st.session_state.master_wav) if st.session_state.master_wav else None
    master_exists = bool(master and master.is_file())
    if controls[3].button(
        "Crear MP3",
        disabled=(
            not master_exists
            or not st.session_state.get("enable_mp3", bool(diagnostics["ffmpeg"]))
            or not bool(diagnostics["ffmpeg"])
            or capabilities.has_active_synthesis
        ),
        use_container_width=True,
    ):
        try:
            assert master is not None
            destination = master.with_suffix(".mp3")
            if destination.exists():
                destination = destination.with_name(f"dialogue-{uuid4().hex[:10]}.mp3")
            export_mp3(master, destination)
            st.session_state.master_mp3 = str(destination)
            st.session_state.project_zip = None
            st.toast("MP3 listo")
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))

    master = Path(st.session_state.master_wav) if st.session_state.master_wav else None
    mp3 = Path(st.session_state.master_mp3) if st.session_state.master_mp3 else None
    if master and master.is_file():
        st.markdown("### Diálogo maestro")
        with st.expander("Escuchar diálogo", expanded=True):
            st.audio(str(master), format="audio/wav")
        downloads = st.columns(2)
        with master.open("rb") as handle:
            downloads[0].download_button(
                "Exportar WAV",
                data=handle,
                file_name=f"{project.title}.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        if mp3 and mp3.is_file():
            with mp3.open("rb") as handle:
                downloads[1].download_button(
                    "Exportar MP3",
                    data=handle,
                    file_name=f"{project.title}.mp3",
                    mime="audio/mpeg",
                    use_container_width=True,
                )
    if st.button(
        "Exportar proyecto ZIP",
        disabled=not capabilities.can_export_project,
        use_container_width=True,
    ):
        try:
            directory = _ensure_saved(store)
            destination = directory / "exports" / f"project-{uuid4().hex[:10]}.zip"
            export_project_zip(
                project,
                directory,
                destination,
                master_wav=master if master and master.is_file() else None,
                master_mp3=mp3 if mp3 and mp3.is_file() else None,
            )
            st.session_state.project_zip = str(destination)
            st.toast("Proyecto portable listo")
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
    project_zip = Path(st.session_state.project_zip) if st.session_state.project_zip else None
    if project_zip and project_zip.is_file():
        with project_zip.open("rb") as handle:
            st.download_button(
                "Descargar proyecto ZIP",
                data=handle,
                file_name=f"{project.title}-proyecto.zip",
                mime="application/zip",
                type="primary",
            )


def main() -> None:
    st.set_page_config(
        page_title="SpeechNote Dialogue Studio",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        paths, store = _init_state()
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"No se pudo preparar la carpeta de trabajo: {exc}")
        st.stop()
    diagnostics = system_diagnostics()
    project: DialogueProject = st.session_state.project
    project_dir = Path(st.session_state.project_dir) if st.session_state.project_dir else None
    capabilities = _project_capabilities(project, diagnostics, project_dir)
    _render_sidebar(paths, store, diagnostics, capabilities)
    _render_header(project, capabilities)
    _render_qwen_backend(paths, diagnostics)
    _render_recovery(project, store, capabilities)
    st.divider()
    _render_speakers(project, paths, diagnostics, capabilities)
    st.divider()
    _render_utterances(project, store, paths, diagnostics, capabilities)
    st.divider()
    _render_global_actions(project, store, paths, diagnostics)
