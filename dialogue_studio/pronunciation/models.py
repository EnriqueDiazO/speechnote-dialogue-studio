"""Validated, serializable pronunciation domain models."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

RuleScope = Literal["builtin", "global", "project", "utterance"]
RuleKind = Literal["literal", "phrase", "acronym", "regex", "math_alias"]
MathStyle = Literal["concise", "classroom", "explicit", "symbolic"]
AcronymPolicy = Literal["custom", "spell_out", "read_as_word", "preserve"]
NumberStyle = Literal["natural", "digits", "preserve"]
UnitStyle = Literal["natural", "spell_out", "preserve"]
PunctuationStyle = Literal["natural", "explicit", "preserve"]

RULE_SCOPES = {"builtin", "global", "project", "utterance"}
RULE_KINDS = {"literal", "phrase", "acronym", "regex", "math_alias"}
MATH_STYLES = {"concise", "classroom", "explicit", "symbolic"}
ACRONYM_POLICIES = {"custom", "spell_out", "read_as_word", "preserve"}
NUMBER_STYLES = {"natural", "digits", "preserve"}
UNIT_STYLES = {"natural", "spell_out", "preserve"}
PUNCTUATION_STYLES = {"natural", "explicit", "preserve"}
MAX_PATTERN_LENGTH = 512
MAX_REPLACEMENT_LENGTH = 2_000
MAX_TEXT_LENGTH = 50_000
_NESTED_QUANTIFIER = re.compile(r"\([^)]*[+*][^)]*\)[+*{]")


def pronunciation_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_language(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("La regla de pronunciación necesita un idioma")
    return normalized


def _validate_regex(pattern: str) -> None:
    if _NESTED_QUANTIFIER.search(pattern):
        raise ValueError("La expresión regular contiene cuantificadores anidados")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Expresión regular inválida: {exc}") from exc


@dataclass(frozen=True)
class PronunciationRule:
    rule_id: str
    scope: RuleScope
    language: str
    kind: RuleKind
    pattern: str
    replacement: str
    enabled: bool = True
    priority: int = 0
    case_sensitive: bool = False
    whole_word: bool = True
    category: str = "custom"
    notes: str = ""
    created_at: str = field(default_factory=pronunciation_now)
    updated_at: str = field(default_factory=pronunciation_now)

    @classmethod
    def create(
        cls,
        *,
        scope: RuleScope,
        language: str,
        kind: RuleKind,
        pattern: str,
        replacement: str,
        priority: int = 0,
        case_sensitive: bool = False,
        whole_word: bool = True,
        category: str = "custom",
        notes: str = "",
        enabled: bool = True,
    ) -> PronunciationRule:
        rule = cls(
            rule_id=str(uuid4()),
            scope=scope,
            language=normalize_language(language),
            kind=kind,
            pattern=unicodedata.normalize("NFC", pattern),
            replacement=unicodedata.normalize("NFC", replacement),
            enabled=enabled,
            priority=priority,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
            category=category.strip() or "custom",
            notes=notes.strip(),
        )
        rule.validate()
        return rule

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        expected_scope: RuleScope | None = None,
    ) -> PronunciationRule:
        scope = str(data.get("scope", expected_scope or "project"))
        rule = cls(
            rule_id=str(data.get("rule_id") or uuid4()),
            scope=scope,  # type: ignore[arg-type]
            language=normalize_language(str(data.get("language", "es"))),
            kind=str(data.get("kind", "literal")),  # type: ignore[arg-type]
            pattern=unicodedata.normalize("NFC", str(data.get("pattern", ""))),
            replacement=unicodedata.normalize("NFC", str(data.get("replacement", ""))),
            enabled=bool(data.get("enabled", True)),
            priority=int(data.get("priority", 0)),
            case_sensitive=bool(data.get("case_sensitive", False)),
            whole_word=bool(data.get("whole_word", True)),
            category=str(data.get("category", "custom")).strip() or "custom",
            notes=str(data.get("notes", "")).strip(),
            created_at=str(data.get("created_at", pronunciation_now())),
            updated_at=str(data.get("updated_at", pronunciation_now())),
        )
        if expected_scope is not None and rule.scope != expected_scope:
            raise ValueError(f"Se esperaba una regla de alcance {expected_scope}")
        rule.validate()
        return rule

    def validate(self) -> None:
        try:
            UUID(self.rule_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("rule_id debe ser un UUID válido") from exc
        if self.scope not in RULE_SCOPES:
            raise ValueError(f"Alcance de pronunciación desconocido: {self.scope}")
        if self.kind not in RULE_KINDS:
            raise ValueError(f"Tipo de regla desconocido: {self.kind}")
        normalize_language(self.language)
        if not self.pattern or len(self.pattern) > MAX_PATTERN_LENGTH:
            raise ValueError("El patrón debe tener entre 1 y 512 caracteres")
        if len(self.replacement) > MAX_REPLACEMENT_LENGTH:
            raise ValueError("La pronunciación es demasiado larga")
        if len(self.category) > 80:
            raise ValueError("La categoría es demasiado larga")
        if not -10_000 <= self.priority <= 10_000:
            raise ValueError("La prioridad debe estar entre -10000 y 10000")
        if self.kind == "regex":
            _validate_regex(self.pattern)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PronunciationProfile:
    enabled: bool = True
    language: str = "es"
    math_style: MathStyle = "classroom"
    acronym_policy: AcronymPolicy = "custom"
    number_style: NumberStyle = "natural"
    unit_style: UnitStyle = "natural"
    punctuation_style: PunctuationStyle = "natural"

    @classmethod
    def for_language(cls, language: str) -> PronunciationProfile:
        normalized = normalize_language(language)
        return cls(language="en" if normalized.startswith("en") else "es")

    @classmethod
    def from_dict(
        cls, data: object, *, fallback_language: str = "es"
    ) -> PronunciationProfile:
        if not isinstance(data, dict):
            return cls.for_language(fallback_language)
        profile = cls(
            enabled=bool(data.get("enabled", True)),
            language=normalize_language(str(data.get("language", fallback_language))),
            math_style=str(data.get("math_style", "classroom")),  # type: ignore[arg-type]
            acronym_policy=str(data.get("acronym_policy", "custom")),  # type: ignore[arg-type]
            number_style=str(data.get("number_style", "natural")),  # type: ignore[arg-type]
            unit_style=str(data.get("unit_style", "natural")),  # type: ignore[arg-type]
            punctuation_style=str(data.get("punctuation_style", "natural")),  # type: ignore[arg-type]
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        normalize_language(self.language)
        if self.math_style not in MATH_STYLES:
            raise ValueError(f"Perfil matemático desconocido: {self.math_style}")
        if self.acronym_policy not in ACRONYM_POLICIES:
            raise ValueError(f"Política de siglas desconocida: {self.acronym_policy}")
        if self.number_style not in NUMBER_STYLES:
            raise ValueError(f"Estilo numérico desconocido: {self.number_style}")
        if self.unit_style not in UNIT_STYLES:
            raise ValueError(f"Estilo de unidades desconocido: {self.unit_style}")
        if self.punctuation_style not in PUNCTUATION_STYLES:
            raise ValueError(f"Estilo de puntuación desconocido: {self.punctuation_style}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AppliedPronunciationRule:
    rule_id: str
    kind: str
    scope: str
    source_span: tuple[int, int]
    source_text: str
    replacement_text: str
    priority: int


@dataclass(frozen=True)
class PronunciationWarning:
    code: str
    message: str
    source_span: tuple[int, int] | None = None
    fragment: str | None = None


@dataclass(frozen=True)
class PronunciationResult:
    written_text: str
    spoken_text: str
    language: str
    profile: PronunciationProfile
    applied_rules: tuple[AppliedPronunciationRule, ...]
    warnings: tuple[PronunciationWarning, ...]
    unsupported_fragments: tuple[str, ...]
    source_hash: str
    rules_hash: str
    engine_version: str
