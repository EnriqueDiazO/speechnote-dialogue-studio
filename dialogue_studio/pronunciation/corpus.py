"""Validated, versioned contracts for pronunciation regression corpora."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .models import MATH_STYLES, PronunciationProfile

CORPUS_SCHEMA_VERSION = 1
CASE_STATUSES = {"approved", "candidate", "deprecated"}
ASSERTION_MODES = {"exact", "semantic", "warning_only"}
SUPPORTED_CORPUS_LANGUAGES = {"es", "en"}
CASE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+){2,}$")
CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

CorpusStatus = Literal["approved", "candidate", "deprecated"]
AssertionMode = Literal["exact", "semantic", "warning_only"]


def _is_nfc(value: str) -> bool:
    return unicodedata.normalize("NFC", value) == value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} debe ser una lista de textos")
    return tuple(value)


@dataclass(frozen=True)
class PronunciationCorpusCase:
    schema_version: int
    case_id: str
    status: CorpusStatus
    language: str
    profile: str
    category: str
    written_text: str
    expected_spoken_text: str
    assertion_mode: AssertionMode = "exact"
    expected_warning_codes: tuple[str, ...] = ()
    expected_unsupported_fragments: tuple[str, ...] = ()
    semantic_anchors: tuple[str, ...] = ()
    forbidden_fragments: tuple[str, ...] = ()
    applied_rule_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str = ""
    source_kind: str = "curated"
    source_reference: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PronunciationCorpusCase:
        case = cls(
            schema_version=int(data.get("schema_version", 0)),
            case_id=str(data.get("case_id", "")),
            status=str(data.get("status", "candidate")),  # type: ignore[arg-type]
            language=str(data.get("language", "")),
            profile=str(data.get("profile", "classroom")),
            category=str(data.get("category", "")),
            written_text=str(data.get("written_text", "")),
            expected_spoken_text=str(data.get("expected_spoken_text", "")),
            assertion_mode=str(data.get("assertion_mode", "exact")),  # type: ignore[arg-type]
            expected_warning_codes=_string_tuple(
                data.get("expected_warning_codes"), "expected_warning_codes"
            ),
            expected_unsupported_fragments=_string_tuple(
                data.get("expected_unsupported_fragments"),
                "expected_unsupported_fragments",
            ),
            semantic_anchors=_string_tuple(data.get("semantic_anchors"), "semantic_anchors"),
            forbidden_fragments=_string_tuple(
                data.get("forbidden_fragments"), "forbidden_fragments"
            ),
            applied_rule_ids=_string_tuple(
                data.get("applied_rule_ids"), "applied_rule_ids"
            ),
            tags=_string_tuple(data.get("tags"), "tags"),
            notes=str(data.get("notes", "")),
            source_kind=str(data.get("source_kind", "curated")),
            source_reference=str(data.get("source_reference", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )
        case.validate()
        return case

    def validate(self) -> None:
        if self.schema_version != CORPUS_SCHEMA_VERSION:
            raise ValueError(f"schema_version de caso no soportada: {self.schema_version}")
        if not CASE_ID_PATTERN.fullmatch(self.case_id):
            raise ValueError(
                "case_id debe ser legible y estable, por ejemplo "
                "es-calculus-derivative-001"
            )
        if self.status not in CASE_STATUSES:
            raise ValueError(f"Estado de corpus desconocido: {self.status}")
        if self.language not in SUPPORTED_CORPUS_LANGUAGES:
            raise ValueError(f"Idioma de corpus no soportado: {self.language}")
        if self.profile not in MATH_STYLES:
            raise ValueError(f"Perfil de corpus no soportado: {self.profile}")
        if not CATEGORY_PATTERN.fullmatch(self.category):
            raise ValueError("La categoría debe ser una clave abierta en snake_case")
        if not self.written_text:
            raise ValueError("written_text es obligatorio")
        if self.assertion_mode not in ASSERTION_MODES:
            raise ValueError(f"Modo de aserción desconocido: {self.assertion_mode}")
        if self.status == "approved" and not self.expected_spoken_text:
            raise ValueError("Un caso approved necesita expected_spoken_text")
        if (
            self.status == "approved"
            and self.assertion_mode == "semantic"
            and not self.semantic_anchors
        ):
            raise ValueError("Un caso semantic aprobado necesita semantic_anchors")
        if (
            self.status == "approved"
            and self.assertion_mode == "warning_only"
            and not self.expected_warning_codes
        ):
            raise ValueError("Un caso warning_only aprobado necesita warnings esperados")
        if not self.source_kind or not CATEGORY_PATTERN.fullmatch(self.source_kind):
            raise ValueError("source_kind debe ser una clave abierta en snake_case")
        if any(
            marker in self.source_reference.casefold()
            for marker in ("/home/", "c:\\users\\", "speech-dialogue-studio/projects/")
        ):
            raise ValueError("source_reference no puede contener rutas personales o de proyectos")
        text_values = (
            self.case_id,
            self.category,
            self.written_text,
            self.expected_spoken_text,
            self.notes,
            self.source_kind,
            self.source_reference,
            *self.expected_warning_codes,
            *self.expected_unsupported_fragments,
            *self.semantic_anchors,
            *self.forbidden_fragments,
            *self.applied_rule_ids,
            *self.tags,
        )
        if not all(_is_nfc(value) for value in text_values):
            raise ValueError("Todos los textos del caso deben estar normalizados a Unicode NFC")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("Los tags no deben repetirse")
        if len(set(self.semantic_anchors)) != len(self.semantic_anchors):
            raise ValueError("Los semantic_anchors no deben repetirse")

    def pronunciation_profile(self) -> PronunciationProfile:
        return PronunciationProfile(
            language=self.language,
            math_style=self.profile,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in (
            "expected_warning_codes",
            "expected_unsupported_fragments",
            "semantic_anchors",
            "forbidden_fragments",
            "applied_rule_ids",
            "tags",
        ):
            data[name] = list(data[name])
        return data


@dataclass(frozen=True)
class PronunciationCorpusManifest:
    schema_version: int
    corpus_version: str
    supported_languages: tuple[str, ...]
    default_profiles: dict[str, str]
    categories: tuple[str, ...]
    case_counts: dict[str, int]
    last_validated_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PronunciationCorpusManifest:
        manifest = cls(
            schema_version=int(data.get("schema_version", 0)),
            corpus_version=str(data.get("corpus_version", "")),
            supported_languages=_string_tuple(
                data.get("supported_languages"), "supported_languages"
            ),
            default_profiles={
                str(key): str(value)
                for key, value in dict(data.get("default_profiles", {})).items()
            },
            categories=_string_tuple(data.get("categories"), "categories"),
            case_counts={
                str(key): int(value)
                for key, value in dict(data.get("case_counts", {})).items()
            },
            last_validated_at=str(data.get("last_validated_at", "")),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != CORPUS_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version de manifest no soportada: {self.schema_version}"
            )
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.corpus_version):
            raise ValueError("corpus_version debe usar formato semántico X.Y.Z")
        if not self.supported_languages:
            raise ValueError("El manifest necesita supported_languages")
        unknown = set(self.supported_languages) - SUPPORTED_CORPUS_LANGUAGES
        if unknown:
            raise ValueError("Idiomas no soportados en manifest: " + ", ".join(sorted(unknown)))
        if set(self.default_profiles) != set(self.supported_languages):
            raise ValueError("default_profiles debe cubrir todos los idiomas soportados")
        if any(profile not in MATH_STYLES for profile in self.default_profiles.values()):
            raise ValueError("El manifest contiene un perfil predeterminado desconocido")
        if not self.categories or any(
            not CATEGORY_PATTERN.fullmatch(category) for category in self.categories
        ):
            raise ValueError("El manifest necesita categorías abiertas válidas")
        required_counts = {"total", "approved", "candidate", "deprecated"}
        if set(self.case_counts) != required_counts:
            raise ValueError("case_counts debe contener total, approved, candidate y deprecated")
        if any(value < 0 for value in self.case_counts.values()):
            raise ValueError("Los conteos del manifest no pueden ser negativos")
        if self.case_counts["total"] != sum(
            self.case_counts[status] for status in ("approved", "candidate", "deprecated")
        ):
            raise ValueError("El total del manifest no coincide con los estados")
        if not self.last_validated_at:
            raise ValueError("last_validated_at debe ser explícito y determinista")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus_version": self.corpus_version,
            "supported_languages": list(self.supported_languages),
            "default_profiles": dict(self.default_profiles),
            "categories": list(self.categories),
            "case_counts": dict(self.case_counts),
            "last_validated_at": self.last_validated_at,
        }


@dataclass(frozen=True)
class PronunciationCorpusSnapshot:
    root: Path
    manifest: PronunciationCorpusManifest
    cases: tuple[PronunciationCorpusCase, ...]
    case_paths: dict[str, Path]

    @property
    def approved(self) -> tuple[PronunciationCorpusCase, ...]:
        return tuple(case for case in self.cases if case.status == "approved")

    @property
    def candidates(self) -> tuple[PronunciationCorpusCase, ...]:
        return tuple(case for case in self.cases if case.status == "candidate")

    @property
    def deprecated(self) -> tuple[PronunciationCorpusCase, ...]:
        return tuple(case for case in self.cases if case.status == "deprecated")


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"El archivo de corpus no es regular: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON de corpus inválido en {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"La raíz JSON debe ser un objeto: {path}")
    return value


def load_pronunciation_corpus(
    root: Path,
    *,
    require_cases: bool = True,
    validate_manifest_counts: bool = True,
) -> PronunciationCorpusSnapshot:
    resolved_root = root.resolve()
    manifest_path = resolved_root / "manifest.json"
    manifest = PronunciationCorpusManifest.from_dict(_load_json_object(manifest_path))
    cases: list[PronunciationCorpusCase] = []
    case_paths: dict[str, Path] = {}
    status_directories = {
        "approved": "approved",
        "candidates": "candidate",
        "deprecated": "deprecated",
    }
    for directory_name, expected_status in status_directories.items():
        directory = resolved_root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Falta el directorio de corpus: {directory_name}")
        for path in sorted(directory.rglob("*.json")):
            if resolved_root not in path.resolve().parents:
                raise ValueError(f"Ruta de corpus fuera de la raíz: {path}")
            relative = path.relative_to(resolved_root)
            if len(relative.parts) < 3:
                raise ValueError(f"El archivo no declara idioma en su ruta: {relative}")
            expected_language = relative.parts[1]
            data = _load_json_object(path)
            if int(data.get("schema_version", 0)) != CORPUS_SCHEMA_VERSION:
                raise ValueError(f"schema_version de archivo no soportada: {relative}")
            raw_cases = data.get("cases")
            if not isinstance(raw_cases, list):
                raise ValueError(f"El archivo necesita una lista cases: {relative}")
            for raw in raw_cases:
                if not isinstance(raw, dict):
                    raise ValueError(f"Caso no válido en {relative}")
                case = PronunciationCorpusCase.from_dict(raw)
                if case.status != expected_status:
                    raise ValueError(
                        f"{case.case_id}: status {case.status} no coincide con {directory_name}"
                    )
                if case.language != expected_language:
                    raise ValueError(
                        f"{case.case_id}: idioma {case.language} no coincide con la ruta"
                    )
                if case.category not in manifest.categories:
                    raise ValueError(
                        f"{case.case_id}: categoría ausente del manifest: {case.category}"
                    )
                if case.case_id in case_paths:
                    first = case_paths[case.case_id].relative_to(resolved_root)
                    raise ValueError(
                        f"case_id duplicado {case.case_id}: {first} y {relative}"
                    )
                cases.append(case)
                case_paths[case.case_id] = path
    cases.sort(key=lambda case: case.case_id)
    if require_cases and not cases:
        raise ValueError("El corpus está vacío")
    counts = Counter(case.status for case in cases)
    actual_counts = {
        "total": len(cases),
        "approved": counts["approved"],
        "candidate": counts["candidate"],
        "deprecated": counts["deprecated"],
    }
    if validate_manifest_counts and manifest.case_counts != actual_counts:
        raise ValueError(
            f"case_counts inconsistente: manifest={manifest.case_counts}, actual={actual_counts}"
        )
    return PronunciationCorpusSnapshot(
        root=resolved_root,
        manifest=manifest,
        cases=tuple(cases),
        case_paths=case_paths,
    )
