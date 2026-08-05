"""Portable domain models for dialogue projects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID, uuid4

SCHEMA_VERSION = 1
UTTERANCE_STATES = {"draft", "generating", "ready", "error", "stale"}
RECOVERABLE_SYNTHESIS_MESSAGE = (
    "La síntesis anterior no terminó. Puedes editar o regenerar esta intervención."
)


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

    @classmethod
    def create(
        cls,
        name: str,
        model_id: str = "",
        model_label: str = "",
        color_key: str = "accent",
    ) -> SpeakerProfile:
        return cls(new_id(), name.strip(), model_id, model_label, color_key, True)

    def validate(self) -> None:
        _valid_uuid(self.speaker_id, "speaker_id")
        if not self.name.strip():
            raise ValueError("Cada hablante necesita un nombre")


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

    @classmethod
    def create(cls, order: int, speaker_id: str, text: str = "") -> Utterance:
        return cls(new_id(), order, speaker_id, text)

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
        speaker_fields = set(SpeakerProfile.__dataclass_fields__)
        utterance_fields = set(Utterance.__dataclass_fields__)
        speakers = [
            SpeakerProfile(**{key: value for key, value in item.items() if key in speaker_fields})
            for item in data.get("speakers", [])
        ]
        utterances = [
            Utterance(**{key: value for key, value in item.items() if key in utterance_fields})
            for item in data.get("utterances", [])
        ]
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
        )
        project.validate()
        return project

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
        if require_utterance and not self.utterances:
            raise ValueError("Se necesita al menos una intervención para exportar")
