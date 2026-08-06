"""Load and validate extensible versioned built-in glossary resources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .models import PronunciationRule

GLOSSARY_VERSION = 1
RESOURCE_ROOT = Path(__file__).parent / "resources"


def _resource_rule(
    *,
    resource_name: str,
    language: str,
    category: str,
    data: dict[str, Any],
    pattern: str,
) -> PronunciationRule:
    kind = str(data.get("kind", "literal"))
    rule_id = str(
        uuid5(
            NAMESPACE_URL,
            f"speechnote-dialogue-studio:{GLOSSARY_VERSION}:{resource_name}:"
            f"{language}:{category}:{kind}:{pattern}",
        )
    )
    rule = PronunciationRule(
        rule_id=rule_id,
        scope="builtin",
        language=language,
        kind=kind,  # type: ignore[arg-type]
        pattern=pattern,
        replacement=str(data.get("replacement", "")),
        enabled=bool(data.get("enabled", True)),
        priority=int(data.get("priority", 0)),
        case_sensitive=bool(data.get("case_sensitive", True)),
        whole_word=bool(data.get("whole_word", True)),
        category=str(data.get("category", category)).strip() or category,
        notes=str(data.get("notes", f"Incorporada · {category}")),
        created_at="2026-08-05T00:00:00+00:00",
        updated_at="2026-08-05T00:00:00+00:00",
    )
    rule.validate()
    return rule


def load_builtin_resource(path: Path) -> list[PronunciationRule]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Recurso de pronunciación inválido: {path.name}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != GLOSSARY_VERSION:
        raise ValueError(f"Versión de recurso no compatible: {path.name}")
    language = str(payload.get("language", "")).strip().lower()
    category = str(payload.get("category", "")).strip()
    entries = payload.get("rules")
    if language not in {"es", "en"} or not category or not isinstance(entries, list):
        raise ValueError(f"Esquema de recurso incompleto: {path.name}")
    rules: list[PronunciationRule] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Regla incorporada inválida: {path.name}")
        raw_patterns = entry.get("patterns", entry.get("pattern"))
        patterns = raw_patterns if isinstance(raw_patterns, list) else [raw_patterns]
        if not patterns or any(not isinstance(pattern, str) for pattern in patterns):
            raise ValueError(f"Patrón incorporado inválido: {path.name}")
        for pattern in patterns:
            rules.append(
                _resource_rule(
                    resource_name=path.name,
                    language=language,
                    category=category,
                    data=entry,
                    pattern=pattern,
                )
            )
    return rules

def builtin_rules(language: str, *, resource_root: Path | None = None) -> list[PronunciationRule]:
    selected = "en" if language.lower().startswith("en") else "es"
    root = (resource_root or RESOURCE_ROOT) / selected
    rules: list[PronunciationRule] = []
    for path in sorted(root.glob("*.json")):
        rules.extend(load_builtin_resource(path))
    if not rules:
        raise ValueError(f"No hay recursos incorporados para el idioma {selected}")
    return rules
