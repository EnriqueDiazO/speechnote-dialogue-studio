"""Versioned dictionary persistence, incremental imports and pending terms."""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from ..paths import AppPaths
from ..storage import atomic_write_text, deterministic_json
from .models import PronunciationRule, RuleScope, pronunciation_now

DICTIONARY_SCHEMA_VERSION = 1
PENDING_SCHEMA_VERSION = 1
ImportMode = Literal["add", "update", "disabled"]
PendingStatus = Literal["pending", "ignored", "ignored_always", "postponed", "added"]


@dataclass(frozen=True)
class DictionaryLoadResult:
    rules: tuple[PronunciationRule, ...]
    warnings: tuple[str, ...] = ()
    recovery_copy: Path | None = None


@dataclass(frozen=True)
class RuleConflict:
    kind: str
    rule_id: str
    related_rule_id: str | None
    message: str


@dataclass(frozen=True)
class RejectedRule:
    index: int
    message: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ImportPreview:
    valid_rules: tuple[PronunciationRule, ...]
    rejected_rules: tuple[RejectedRule, ...]
    conflicts: tuple[RuleConflict, ...]


@dataclass(frozen=True)
class PendingTerm:
    candidate_id: str
    term: str
    language: str
    context: str
    source: str
    project_id: str | None = None
    utterance_id: str | None = None
    category: str = "technical"
    occurrences: int = 1
    status: PendingStatus = "pending"
    created_at: str = field(default_factory=pronunciation_now)
    updated_at: str = field(default_factory=pronunciation_now)

    @classmethod
    def create(
        cls,
        term: str,
        *,
        language: str,
        context: str,
        source: str,
        project_id: str | None = None,
        utterance_id: str | None = None,
        category: str = "technical",
    ) -> PendingTerm:
        candidate = cls(
            candidate_id=str(uuid4()),
            term=term.strip(),
            language=language.strip().lower(),
            context=context.strip()[:1_000],
            source=source.strip()[:80],
            project_id=project_id,
            utterance_id=utterance_id,
            category=category.strip()[:80] or "technical",
        )
        candidate.validate()
        return candidate

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingTerm:
        candidate = cls(
            candidate_id=str(data["candidate_id"]),
            term=str(data.get("term", "")),
            language=str(data.get("language", "es")),
            context=str(data.get("context", "")),
            source=str(data.get("source", "unknown")),
            project_id=(str(data["project_id"]) if data.get("project_id") else None),
            utterance_id=(
                str(data["utterance_id"]) if data.get("utterance_id") else None
            ),
            category=str(data.get("category", "technical")),
            occurrences=int(data.get("occurrences", 1)),
            status=str(data.get("status", "pending")),  # type: ignore[arg-type]
            created_at=str(data.get("created_at", pronunciation_now())),
            updated_at=str(data.get("updated_at", pronunciation_now())),
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        UUID(self.candidate_id)
        if not self.term or len(self.term) > 512:
            raise ValueError("El término candidato debe tener entre 1 y 512 caracteres")
        if self.status not in {"pending", "ignored", "ignored_always", "postponed", "added"}:
            raise ValueError("Estado de término pendiente desconocido")
        if self.occurrences < 1:
            raise ValueError("El contador del término debe ser positivo")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "term": self.term,
            "language": self.language,
            "context": self.context,
            "source": self.source,
            "project_id": self.project_id,
            "utterance_id": self.utterance_id,
            "category": self.category,
            "occurrences": self.occurrences,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _recovery_copy(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.corrupt-{stamp}.bak")
    counter = 1
    while destination.exists():
        destination = path.with_name(f"{path.name}.corrupt-{stamp}-{counter}.bak")
        counter += 1
    shutil.copy2(path, destination)
    return destination


class GlobalPronunciationStore:
    def __init__(self, paths: AppPaths):
        self.paths = paths
        self.paths.ensure()

    def load(self) -> DictionaryLoadResult:
        path = self.paths.pronunciation_dictionary
        if not path.exists():
            return DictionaryLoadResult(())
        if path.is_symlink():
            return DictionaryLoadResult(
                (),
                ("El diccionario global no puede ser un enlace simbólico",),
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("La raíz del diccionario debe ser un objeto JSON")
            version = int(data.get("schema_version", 1))
            if version != DICTIONARY_SCHEMA_VERSION:
                raise ValueError(f"schema_version de diccionario no soportada: {version}")
            rules = tuple(
                PronunciationRule.from_dict(item, expected_scope="global")
                for item in data.get("rules", [])
                if isinstance(item, dict)
            )
            return DictionaryLoadResult(rules)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            recovery = _recovery_copy(path)
            return DictionaryLoadResult(
                (),
                (f"No se pudo cargar el diccionario global: {exc}",),
                recovery,
            )

    def save(self, rules: list[PronunciationRule] | tuple[PronunciationRule, ...]) -> Path:
        for rule in rules:
            rule.validate()
            if rule.scope != "global":
                raise ValueError("El diccionario global sólo admite reglas globales")
        payload = {
            "schema_version": DICTIONARY_SCHEMA_VERSION,
            "rules": [rule.to_dict() for rule in rules],
        }
        atomic_write_text(self.paths.pronunciation_dictionary, deterministic_json(payload))
        return self.paths.pronunciation_dictionary


class PendingTermStore:
    def __init__(self, paths: AppPaths):
        self.paths = paths
        self.paths.ensure()

    def load(self) -> tuple[PendingTerm, ...]:
        path = self.paths.pronunciation_pending_terms
        if not path.exists():
            return ()
        if path.is_symlink():
            return ()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or int(data.get("schema_version", 1)) != 1:
                raise ValueError("Formato de bandeja de términos no soportado")
            return tuple(
                PendingTerm.from_dict(item)
                for item in data.get("terms", [])
                if isinstance(item, dict)
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            _recovery_copy(path)
            return ()

    def save(self, terms: list[PendingTerm] | tuple[PendingTerm, ...]) -> Path:
        for term in terms:
            term.validate()
        payload = {
            "schema_version": PENDING_SCHEMA_VERSION,
            "terms": [term.to_dict() for term in terms],
        }
        atomic_write_text(self.paths.pronunciation_pending_terms, deterministic_json(payload))
        return self.paths.pronunciation_pending_terms

    def record(self, candidate: PendingTerm) -> tuple[PendingTerm, ...]:
        terms = list(self.load())
        identity = (candidate.term.casefold(), candidate.language.split("-", 1)[0])
        for index, term in enumerate(terms):
            other = (term.term.casefold(), term.language.split("-", 1)[0])
            if other != identity:
                continue
            if term.status == "ignored_always":
                return tuple(terms)
            terms[index] = replace(
                term,
                occurrences=term.occurrences + 1,
                context=candidate.context or term.context,
                updated_at=pronunciation_now(),
            )
            self.save(terms)
            return tuple(terms)
        terms.append(candidate)
        self.save(terms)
        return tuple(terms)

    def set_status(self, candidate_id: str, status: PendingStatus) -> tuple[PendingTerm, ...]:
        terms = list(self.load())
        found = False
        for index, term in enumerate(terms):
            if term.candidate_id == candidate_id:
                terms[index] = replace(term, status=status, updated_at=pronunciation_now())
                found = True
                break
        if not found:
            raise ValueError("No se encontró el término pendiente")
        self.save(terms)
        return tuple(terms)


def _same_match(left: PronunciationRule, right: PronunciationRule) -> bool:
    return (
        left.scope,
        left.language,
        left.kind,
        left.pattern if left.case_sensitive else left.pattern.casefold(),
        left.case_sensitive,
        left.whole_word,
    ) == (
        right.scope,
        right.language,
        right.kind,
        right.pattern if right.case_sensitive else right.pattern.casefold(),
        right.case_sensitive,
        right.whole_word,
    )


def detect_rule_conflicts(
    candidate: PronunciationRule,
    existing: list[PronunciationRule] | tuple[PronunciationRule, ...],
) -> tuple[RuleConflict, ...]:
    conflicts: list[RuleConflict] = []
    for rule in existing:
        if rule.rule_id == candidate.rule_id:
            conflicts.append(
                RuleConflict("rule_id", candidate.rule_id, rule.rule_id, "El rule_id ya existe")
            )
        if _same_match(candidate, rule):
            if candidate.replacement == rule.replacement:
                message = "La misma regla ya existe"
                kind = "duplicate"
            else:
                message = "Otra regla coincide con el mismo patrón y puede ensombrecerla"
                kind = "shadowing"
            conflicts.append(RuleConflict(kind, candidate.rule_id, rule.rule_id, message))
        if (
            candidate.kind != "regex"
            and rule.kind != "regex"
            and candidate.pattern.casefold() == rule.replacement.casefold()
            and candidate.replacement.casefold() == rule.pattern.casefold()
        ):
            conflicts.append(
                RuleConflict(
                    "cycle",
                    candidate.rule_id,
                    rule.rule_id,
                    "La regla forma un ciclo simple de reemplazo",
                )
            )
    if candidate.kind == "regex":
        groups = re.compile(candidate.pattern).groups
        references = {
            int(first or second)
            for first, second in re.findall(r"\\g<(\d+)>|\\(\d+)", candidate.replacement)
        }
        if any(reference > groups for reference in references):
            conflicts.append(
                RuleConflict(
                    "regex_group",
                    candidate.rule_id,
                    None,
                    "El reemplazo referencia un grupo regex inexistente",
                )
            )
    return tuple(conflicts)


def export_rules_json(rules: list[PronunciationRule] | tuple[PronunciationRule, ...]) -> str:
    return deterministic_json(
        {
            "schema_version": DICTIONARY_SCHEMA_VERSION,
            "rules": [rule.to_dict() for rule in rules],
        }
    )


def export_rules_csv(rules: list[PronunciationRule] | tuple[PronunciationRule, ...]) -> str:
    output = io.StringIO()
    fields = (
        "rule_id", "scope", "language", "kind", "pattern", "replacement", "enabled",
        "priority", "case_sensitive", "whole_word", "category", "notes", "created_at",
        "updated_at", "usage_count", "last_used_at",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for rule in rules:
        data = rule.to_dict()
        writer.writerow({name: data.get(name, "") for name in fields})
    return output.getvalue()


def _csv_bool(value: object, default: bool) -> bool:
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "sí", "si"}


def preview_rule_import(
    content: str,
    *,
    format_name: Literal["json", "csv"],
    scope: RuleScope,
    existing: list[PronunciationRule] | tuple[PronunciationRule, ...] = (),
) -> ImportPreview:
    if format_name == "json":
        try:
            root = json.loads(content)
        except json.JSONDecodeError as exc:
            return ImportPreview(
                (),
                (RejectedRule(0, f"JSON corrupto: {exc}", {}),),
                (),
            )
        raw_rules = root.get("rules", []) if isinstance(root, dict) else root
        if not isinstance(raw_rules, list):
            raise ValueError("El JSON importado necesita una lista de rules")
    else:
        raw_rules = list(csv.DictReader(io.StringIO(content)))
    valid: list[PronunciationRule] = []
    rejected: list[RejectedRule] = []
    conflicts: list[RuleConflict] = []
    for index, raw_value in enumerate(raw_rules, start=1):
        raw = dict(raw_value) if isinstance(raw_value, dict) else {"value": raw_value}
        try:
            if format_name == "csv":
                raw["enabled"] = _csv_bool(raw.get("enabled"), True)
                raw["case_sensitive"] = _csv_bool(raw.get("case_sensitive"), False)
                raw["whole_word"] = _csv_bool(raw.get("whole_word"), True)
                raw["priority"] = int(raw.get("priority") or 0)
                raw["usage_count"] = int(raw.get("usage_count") or 0)
            raw.setdefault("scope", scope)
            rule = PronunciationRule.from_dict(raw, expected_scope=scope)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(RejectedRule(index, str(exc), raw))
            continue
        conflicts.extend(detect_rule_conflicts(rule, [*existing, *valid]))
        valid.append(rule)
    return ImportPreview(tuple(valid), tuple(rejected), tuple(conflicts))


def merge_imported_rules(
    existing: list[PronunciationRule] | tuple[PronunciationRule, ...],
    imported: list[PronunciationRule] | tuple[PronunciationRule, ...],
    *,
    mode: ImportMode,
) -> tuple[PronunciationRule, ...]:
    result = list(existing)
    positions = {rule.rule_id: index for index, rule in enumerate(result)}
    signatures = {
        (rule.scope, rule.language, rule.kind, rule.pattern.casefold(), rule.replacement.casefold())
        for rule in result
    }
    for rule in imported:
        candidate = replace(rule, enabled=False) if mode == "disabled" else rule
        if mode == "update" and candidate.rule_id in positions:
            result[positions[candidate.rule_id]] = candidate
            continue
        signature = (
            candidate.scope,
            candidate.language,
            candidate.kind,
            candidate.pattern.casefold(),
            candidate.replacement.casefold(),
        )
        if candidate.rule_id in positions or signature in signatures:
            continue
        positions[candidate.rule_id] = len(result)
        signatures.add(signature)
        result.append(candidate)
    return tuple(result)


def update_rule_with_audit(
    rule: PronunciationRule,
    *,
    changes: dict[str, Any],
    actor: str = "user",
    context: str = "ui",
) -> PronunciationRule:
    allowed = {
        "language", "kind", "pattern", "replacement", "enabled", "priority",
        "case_sensitive", "whole_word", "category", "notes",
    }
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError("Campos de regla no editables: " + ", ".join(sorted(unknown)))
    now = pronunciation_now()
    event = {
        "changed_at": now,
        "actor": actor,
        "context": context,
        "fields": ",".join(sorted(changes)),
    }
    updated = replace(
        rule,
        **changes,
        updated_at=now,
        change_history=(*rule.change_history, event),
    )
    updated.validate()
    return updated


def record_rule_usage(
    rules: list[PronunciationRule] | tuple[PronunciationRule, ...],
    applied_rule_ids: set[str],
) -> tuple[PronunciationRule, ...]:
    now = pronunciation_now()
    return tuple(
        replace(rule, usage_count=rule.usage_count + 1, last_used_at=now)
        if rule.rule_id in applied_rule_ids
        else rule
        for rule in rules
    )
