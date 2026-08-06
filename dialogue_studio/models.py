"""Portable domain models for dialogue projects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal
from uuid import UUID, uuid4

from .pronunciation import PronunciationProfile, PronunciationRule

SCHEMA_VERSION = 1
UTTERANCE_STATES = {"draft", "generating", "ready", "error", "stale"}
RECOVERABLE_SYNTHESIS_MESSAGE = (
    "La síntesis anterior no terminó. Puedes editar o regenerar esta intervención."
)
TTSProvider = Literal["speechnote", "qwen"]
TTS_PROVIDERS = {"speechnote", "qwen"}


@dataclass
class SpeakerTTSConfig:
    """Durable provider settings for one character.

    ``instruction_text`` is reserved for future models that advertise ``supports_instruct``.
    It remains absent from serialized 0.6B configurations when unset.
    """

    provider: TTSProvider = "speechnote"
    voice_id: str = ""
    voice_label: str = ""
    language: str = "auto"
    generation_options: dict[str, int | float] = field(default_factory=dict)
    instruction_text: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        legacy_voice_id: str = "",
        legacy_voice_label: str = "",
    ) -> SpeakerTTSConfig:
        if not isinstance(data, dict):
            return cls(
                provider="speechnote",
                voice_id=legacy_voice_id,
                voice_label=legacy_voice_label,
            )
        return cls(
            provider=str(data.get("provider", "speechnote")),  # type: ignore[arg-type]
            voice_id=str(data.get("voice_id", legacy_voice_id)),
            voice_label=str(data.get("voice_label", legacy_voice_label)),
            language=str(data.get("language", "auto")),
            generation_options=dict(data.get("generation_options", {})),
            instruction_text=(
                str(data["instruction_text"]) if data.get("instruction_text") else None
            ),
        )

    def validate(self) -> None:
        if self.provider not in TTS_PROVIDERS:
            raise ValueError(f"Proveedor TTS desconocido: {self.provider}")
        if not self.language.strip():
            raise ValueError("La configuración TTS necesita un idioma")
        for name, value in self.generation_options.items():
            if not isinstance(name, str) or isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError("Las opciones TTS deben ser valores numéricos")


@dataclass
class UtteranceTTSOverride:
    provider: TTSProvider | None = None
    voice_id: str | None = None
    language: str | None = None
    generation_options: dict[str, int | float] = field(default_factory=dict)
    instruction_text: str | None = None

    @classmethod
    def from_dict(cls, data: object) -> UtteranceTTSOverride | None:
        if not isinstance(data, dict):
            return None
        override = cls(
            provider=(str(data["provider"]) if data.get("provider") else None),  # type: ignore[arg-type]
            voice_id=(str(data["voice_id"]) if data.get("voice_id") else None),
            language=(str(data["language"]) if data.get("language") else None),
            generation_options=dict(data.get("generation_options", {})),
            instruction_text=(
                str(data["instruction_text"]) if data.get("instruction_text") else None
            ),
        )
        return None if override.is_empty else override

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.provider,
                self.voice_id,
                self.language,
                self.generation_options,
                self.instruction_text,
            )
        )

    def validate(self) -> None:
        if self.provider is not None and self.provider not in TTS_PROVIDERS:
            raise ValueError(f"Proveedor TTS desconocido: {self.provider}")
        if self.voice_id is not None and not self.voice_id.strip():
            raise ValueError("El override de voz no puede estar vacío")
        if self.language is not None and not self.language.strip():
            raise ValueError("El override de idioma no puede estar vacío")
        for name, value in self.generation_options.items():
            if not isinstance(name, str) or isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError("Las opciones TTS deben ser valores numéricos")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return str(uuid4())


def _valid_uuid(value: str, label: str) -> None:
    try:
        UUID(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} debe ser un UUID válido") from exc


def validate_relative_path(value: str | None) -> None:
    if not value:
        return
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Las rutas de audio deben ser relativas y portables")


@dataclass
class SpeakerProfile:
    speaker_id: str
    name: str
    model_id: str
    model_label: str
    color_key: str = "accent"
    enabled: bool = True
    tts: SpeakerTTSConfig | None = None

    @classmethod
    def create(
        cls,
        name: str,
        model_id: str = "",
        model_label: str = "",
        color_key: str = "accent",
        *,
        tts: SpeakerTTSConfig | None = None,
    ) -> SpeakerProfile:
        config = tts or SpeakerTTSConfig(
            provider="speechnote",
            voice_id=model_id,
            voice_label=model_label,
        )
        return cls(new_id(), name.strip(), model_id, model_label, color_key, True, config)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeakerProfile:
        model_id = str(data.get("model_id", ""))
        model_label = str(data.get("model_label", ""))
        return cls(
            speaker_id=str(data["speaker_id"]),
            name=str(data.get("name", "")),
            model_id=model_id,
            model_label=model_label,
            color_key=str(data.get("color_key", "accent")),
            enabled=bool(data.get("enabled", True)),
            tts=SpeakerTTSConfig.from_dict(
                data.get("tts"),
                legacy_voice_id=model_id,
                legacy_voice_label=model_label,
            ),
        )

    @property
    def tts_config(self) -> SpeakerTTSConfig:
        if self.tts is None:
            self.tts = SpeakerTTSConfig(
                provider="speechnote",
                voice_id=self.model_id,
                voice_label=self.model_label,
            )
        return self.tts

    def validate(self) -> None:
        _valid_uuid(self.speaker_id, "speaker_id")
        if not self.name.strip():
            raise ValueError("Cada hablante necesita un nombre")
        self.tts_config.validate()


@dataclass
class Utterance:
    utterance_id: str
    order: int
    speaker_id: str
    text: str = ""
    audio_relative_path: str | None = None
    duration_seconds: float | None = None
    sha256: str | None = None
    status: str = "draft"
    error_message: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    tts_override: UtteranceTTSOverride | None = None
    audio_fingerprint: str | None = None
    use_pronunciation_engine: bool = True
    manual_spoken_text_override: str | None = None
    utterance_rules: list[PronunciationRule] = field(default_factory=list)
    spoken_text: str | None = None
    written_text_hash: str | None = None
    spoken_text_hash: str | None = None
    pronunciation_rules_hash: str | None = None
    pronunciation_engine_version: str | None = None
    applied_pronunciation_rule_ids: list[str] = field(default_factory=list)
    pronunciation_warnings: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, order: int, speaker_id: str, text: str = "") -> Utterance:
        return cls(new_id(), order, speaker_id, text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Utterance:
        fields = set(cls.__dataclass_fields__)
        values = {key: value for key, value in data.items() if key in fields}
        if "text" not in values and "written_text" in data:
            values["text"] = str(data["written_text"])
        values["tts_override"] = UtteranceTTSOverride.from_dict(data.get("tts_override"))
        values["utterance_rules"] = [
            PronunciationRule.from_dict(item, expected_scope="utterance")
            for item in data.get("utterance_rules", [])
            if isinstance(item, dict)
        ]
        return cls(**values)

    def validate(self) -> None:
        _valid_uuid(self.utterance_id, "utterance_id")
        _valid_uuid(self.speaker_id, "speaker_id")
        if self.order < 1:
            raise ValueError("El orden de una intervención debe ser positivo")
        if self.status not in UTTERANCE_STATES:
            raise ValueError(f"Estado de intervención desconocido: {self.status}")
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("La duración no puede ser negativa")
        validate_relative_path(self.audio_relative_path)
        if self.tts_override is not None:
            self.tts_override.validate()
        for rule in self.utterance_rules:
            rule.validate()
            if rule.scope != "utterance":
                raise ValueError("Las reglas de intervención necesitan alcance utterance")

    def mark_stale(self) -> None:
        if self.audio_relative_path or self.status == "ready":
            self.status = "stale"
        elif self.status != "generating":
            self.status = "draft"
        self.error_message = None
        self.updated_at = utc_now()


@dataclass
class DialogueProject:
    schema_version: int
    project_id: str
    title: str
    description: str
    language: str
    pause_ms: int
    speakers: list[SpeakerProfile]
    utterances: list[Utterance]
    created_at: str
    updated_at: str
    pronunciation_profile: PronunciationProfile = field(
        default_factory=PronunciationProfile
    )
    pronunciation_rules: list[PronunciationRule] = field(default_factory=list)

    @classmethod
    def new(cls, title: str = "Diálogo sin título") -> DialogueProject:
        now = utc_now()
        professor = SpeakerProfile.create(
            "Profesor",
            "es_piper_mx_claude_high",
            "Español mexicano · Piper Claude High",
            "professor",
        )
        student = SpeakerProfile.create(
            "Estudiante",
            "es_piper_es_sharvard_medium_1",
            "Español · Piper Sharvard Medium Female",
            "student",
        )
        return cls(
            schema_version=SCHEMA_VERSION,
            project_id=new_id(),
            title=title,
            description="",
            language="es-MX",
            pause_ms=650,
            speakers=[professor, student],
            utterances=[Utterance.create(1, professor.speaker_id)],
            created_at=now,
            updated_at=now,
            pronunciation_profile=PronunciationProfile.for_language("es-MX"),
        )

    @classmethod
    def sample(cls) -> DialogueProject:
        project = cls.new("Cómo aprende una red neuronal")
        professor, student = project.speakers
        project.description = "Diálogo breve de ejemplo para comprobar las voces."
        project.utterances = [
            Utterance.create(
                1,
                professor.speaker_id,
                "Hoy estudiaremos cómo aprende una red neuronal artificial.",
            ),
            Utterance.create(
                2,
                student.speaker_id,
                "¿El aprendizaje consiste únicamente en modificar los pesos de la red?",
            ),
            Utterance.create(
                3,
                professor.speaker_id,
                "No solamente. También se modifican los sesgos para reducir la función de pérdida.",
            ),
        ]
        project.touch()
        return project

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DialogueProject:
        version = int(data.get("schema_version", 1))
        if version > SCHEMA_VERSION:
            raise ValueError(
                "El proyecto usa schema_version "
                f"{version}; esta versión admite hasta {SCHEMA_VERSION}"
            )
        speakers = [SpeakerProfile.from_dict(item) for item in data.get("speakers", [])]
        utterances = [Utterance.from_dict(item) for item in data.get("utterances", [])]
        project = cls(
            schema_version=version,
            project_id=data["project_id"],
            title=data.get("title", "Diálogo sin título"),
            description=data.get("description", ""),
            language=data.get("language", "es-MX"),
            pause_ms=int(data.get("pause_ms", 650)),
            speakers=speakers,
            utterances=utterances,
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
            pronunciation_profile=PronunciationProfile.from_dict(
                data.get("pronunciation_profile"),
                fallback_language=str(data.get("language", "es-MX")),
            ),
            pronunciation_rules=[
                PronunciationRule.from_dict(item, expected_scope="project")
                for item in data.get("pronunciation_rules", [])
                if isinstance(item, dict)
            ],
        )
        project.validate()
        return project

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for speaker in data["speakers"]:
            tts = speaker.get("tts")
            if isinstance(tts, dict) and tts.get("instruction_text") is None:
                tts.pop("instruction_text", None)
        for utterance in data["utterances"]:
            utterance["written_text"] = utterance["text"]
            override = utterance.get("tts_override")
            if override is None:
                utterance.pop("tts_override", None)
            elif isinstance(override, dict):
                for key in ("provider", "voice_id", "language", "instruction_text"):
                    if override.get(key) is None:
                        override.pop(key, None)
                if not override.get("generation_options"):
                    override.pop("generation_options", None)
            if utterance.get("audio_fingerprint") is None:
                utterance.pop("audio_fingerprint", None)
        return data

    def touch(self) -> None:
        self.updated_at = utc_now()

    def normalize_order(self) -> None:
        for index, utterance in enumerate(self.utterances, start=1):
            utterance.order = index
        self.touch()

    def speaker(self, speaker_id: str) -> SpeakerProfile:
        for speaker in self.speakers:
            if speaker.speaker_id == speaker_id:
                return speaker
        raise ValueError("La intervención referencia un hablante inexistente")

    def validate(self, *, require_utterance: bool = False) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version no soportada: {self.schema_version}")
        _valid_uuid(self.project_id, "project_id")
        if not self.speakers:
            raise ValueError("El proyecto necesita al menos un hablante")
        if not 0 <= self.pause_ms <= 5000:
            raise ValueError("pause_ms debe estar entre 0 y 5000")
        for speaker in self.speakers:
            speaker.validate()
        known_speakers = {speaker.speaker_id for speaker in self.speakers}
        expected_orders = list(range(1, len(self.utterances) + 1))
        if [item.order for item in self.utterances] != expected_orders:
            raise ValueError("El orden de las intervenciones debe ser consecutivo")
        for utterance in self.utterances:
            utterance.validate()
            if utterance.speaker_id not in known_speakers:
                raise ValueError("Una intervención referencia un hablante inexistente")
        self.pronunciation_profile.validate()
        for rule in self.pronunciation_rules:
            rule.validate()
            if rule.scope != "project":
                raise ValueError("Las reglas del proyecto necesitan alcance project")
        if require_utterance and not self.utterances:
            raise ValueError("Se necesita al menos una intervención para exportar")


def effective_tts_config(
    project: DialogueProject, utterance: Utterance
) -> SpeakerTTSConfig:
    """Resolve a durable character configuration plus an optional utterance override."""
    base = project.speaker(utterance.speaker_id).tts_config
    override = utterance.tts_override
    if override is None:
        return SpeakerTTSConfig(
            provider=base.provider,
            voice_id=base.voice_id,
            voice_label=base.voice_label,
            language=base.language,
            generation_options=dict(base.generation_options),
            instruction_text=base.instruction_text,
        )
    options = dict(base.generation_options)
    options.update(override.generation_options)
    return SpeakerTTSConfig(
        provider=override.provider or base.provider,
        voice_id=override.voice_id or base.voice_id,
        voice_label=(
            override.voice_id.replace("_", " ").title()
            if override.voice_id
            else base.voice_label
        ),
        language=override.language or base.language,
        generation_options=options,
        instruction_text=(
            override.instruction_text
            if override.instruction_text is not None
            else base.instruction_text
        ),
    )
