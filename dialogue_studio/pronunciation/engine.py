"""Deterministic provider-neutral pronunciation rule engine."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass

from .linguistics import apply_linguistic_styles
from .math_speech import verbalize_math
from .models import (
    MAX_TEXT_LENGTH,
    AppliedPronunciationRule,
    PronunciationProfile,
    PronunciationResult,
    PronunciationRule,
    PronunciationWarning,
)
from .segmentation import PROTECTED_SEGMENTS, segment_text

ENGINE_VERSION = "1.0"
SCOPE_RANK = {"builtin": 0, "global": 1, "project": 2, "utterance": 3}


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    replacement: str
    rule: PronunciationRule


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rules_hash(rules: list[PronunciationRule]) -> str:
    active = [rule.behavior_dict() for rule in rules if rule.enabled]
    active.sort(key=lambda item: str(item["rule_id"]))
    return _canonical_hash(active)


def detect_rule_cycles(rules: list[PronunciationRule]) -> set[str]:
    literal = {
        rule.pattern.casefold(): (rule.replacement.casefold(), rule.rule_id)
        for rule in rules
        if rule.enabled
        and rule.kind != "regex"
        and rule.pattern.casefold() != rule.replacement.casefold()
    }
    cyclic: set[str] = set()
    for source, (target, rule_id) in literal.items():
        reverse = literal.get(target)
        if reverse is not None and reverse[0] == source:
            cyclic.update((rule_id, reverse[1]))
    return cyclic


def _language_matches(rule: PronunciationRule, language: str) -> bool:
    rule_language = rule.language.split("-", 1)[0]
    target_language = language.split("-", 1)[0]
    return rule_language in {"*", target_language}


def _rule_pattern(rule: PronunciationRule) -> re.Pattern[str]:
    flags = 0 if rule.case_sensitive else re.IGNORECASE
    pattern = rule.pattern if rule.kind == "regex" else re.escape(rule.pattern)
    if rule.whole_word:
        pattern = rf"(?<!\w)(?:{pattern})(?!\w)"
    return re.compile(pattern, flags)


def _candidate_sort_key(candidate: _Candidate) -> tuple[int, int, int, str]:
    return (
        -SCOPE_RANK[candidate.rule.scope],
        -candidate.rule.priority,
        -(candidate.end - candidate.start),
        candidate.rule.rule_id,
    )


def _apply_rules_once(
    text: str,
    rules: list[PronunciationRule],
    language: str,
    *,
    offset: int = 0,
) -> tuple[str, list[AppliedPronunciationRule]]:
    candidates: list[_Candidate] = []
    for rule in rules:
        if not rule.enabled or not _language_matches(rule, language):
            continue
        for match in _rule_pattern(rule).finditer(text):
            try:
                replacement = (
                    match.expand(rule.replacement)
                    if rule.kind == "regex"
                    else rule.replacement
                )
            except re.error:
                replacement = rule.replacement
            candidates.append(_Candidate(match.start(), match.end(), replacement, rule))
    by_start: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        by_start.setdefault(candidate.start, []).append(candidate)
    output: list[str] = []
    applied: list[AppliedPronunciationRule] = []
    cursor = 0
    while cursor < len(text):
        available = [item for item in by_start.get(cursor, []) if item.end > cursor]
        if not available:
            output.append(text[cursor])
            cursor += 1
            continue
        selected = sorted(available, key=_candidate_sort_key)[0]
        source = text[selected.start : selected.end]
        output.append(selected.replacement)
        applied.append(
            AppliedPronunciationRule(
                rule_id=selected.rule.rule_id,
                kind=selected.rule.kind,
                scope=selected.rule.scope,
                source_span=(offset + selected.start, offset + selected.end),
                source_text=source,
                replacement_text=selected.replacement,
                priority=selected.rule.priority,
            )
        )
        cursor = selected.end
    return "".join(output), applied


class PronunciationEngine:
    version = ENGINE_VERSION

    def transform(
        self,
        written_text: str,
        *,
        profile: PronunciationProfile | None = None,
        rules: list[PronunciationRule] | None = None,
        manual_override: str | None = None,
    ) -> PronunciationResult:
        selected_profile = profile or PronunciationProfile()
        selected_profile.validate()
        normalized = unicodedata.normalize("NFC", written_text)
        if len(normalized) > MAX_TEXT_LENGTH:
            raise ValueError(f"El texto supera el límite de {MAX_TEXT_LENGTH} caracteres")
        from .glossary import builtin_rules

        selected_rules = [*builtin_rules(selected_profile.language), *(rules or [])]
        for rule in selected_rules:
            rule.validate()
        digest = rules_hash(selected_rules)
        source_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        warnings: list[PronunciationWarning] = []
        cyclic_ids = detect_rule_cycles(selected_rules)
        if cyclic_ids:
            warnings.append(
                PronunciationWarning(
                    code="rule_cycle",
                    message=(
                        "Se omitieron reglas con un ciclo simple de reemplazo: "
                        + ", ".join(sorted(cyclic_ids))
                    ),
                )
            )
            selected_rules = [rule for rule in selected_rules if rule.rule_id not in cyclic_ids]
        if manual_override is not None:
            spoken = unicodedata.normalize("NFC", manual_override).strip()
            if not spoken:
                warnings.append(
                    PronunciationWarning(
                        code="empty_manual_override",
                        message="El override manual está vacío; se usó el texto original.",
                    )
                )
                spoken = normalized
            applied = (
                AppliedPronunciationRule(
                    rule_id="manual-utterance-override",
                    kind="manual_override",
                    scope="utterance",
                    source_span=(0, len(normalized)),
                    source_text=normalized,
                    replacement_text=spoken,
                    priority=10_001,
                ),
            )
        elif not selected_profile.enabled:
            spoken = normalized
            applied = ()
        else:
            chunks: list[str] = []
            applied_list: list[AppliedPronunciationRule] = []
            unsupported: list[str] = []
            for segment in segment_text(normalized):
                available_rules = selected_rules
                if segment.kind in PROTECTED_SEGMENTS:
                    available_rules = [
                        rule for rule in selected_rules if rule.scope == "utterance"
                    ]
                elif segment.kind.startswith("math_"):
                    available_rules = [
                        rule
                        for rule in selected_rules
                        if rule.scope != "builtin" and rule.kind != "math_alias"
                    ]
                elif segment.kind == "prose":
                    available_rules = [
                        rule
                        for rule in selected_rules
                        if rule.kind != "math_alias" and rule.category != "unit"
                    ]
                transformed, segment_applied = _apply_rules_once(
                    segment.text,
                    available_rules,
                    selected_profile.language,
                    offset=segment.start,
                )
                if segment.kind.startswith("math_"):
                    aliases = {
                        rule.pattern: rule.replacement
                        for rule in selected_rules
                        if rule.enabled
                        and rule.kind == "math_alias"
                        and _language_matches(rule, selected_profile.language)
                    }
                    math_result = verbalize_math(
                        transformed,
                        language=selected_profile.language,
                        style=selected_profile.math_style,
                        number_style=selected_profile.number_style,
                        aliases=aliases,
                        source_offset=segment.start,
                    )
                    transformed = math_result.spoken_text
                    warnings.extend(math_result.warnings)
                    unsupported.extend(math_result.unsupported_fragments)
                    applied_list.append(
                        AppliedPronunciationRule(
                            rule_id=f"builtin-math-{selected_profile.math_style}",
                            kind="math",
                            scope="builtin",
                            source_span=(segment.start, segment.end),
                            source_text=segment.text,
                            replacement_text=transformed,
                            priority=-100,
                        )
                    )
                elif segment.kind == "prose":
                    known = {
                        rule.pattern
                        for rule in selected_rules
                        if rule.enabled and rule.kind == "acronym"
                    }
                    units = {
                        rule.pattern: rule.replacement
                        for rule in selected_rules
                        if rule.enabled
                        and rule.category == "unit"
                        and _language_matches(rule, selected_profile.language)
                    }
                    transformed, linguistic_applied, linguistic_warnings, candidates = (
                        apply_linguistic_styles(
                            transformed,
                            selected_profile,
                            known_acronyms=known,
                            units=units,
                            source_offset=segment.start,
                        )
                    )
                    applied_list.extend(linguistic_applied)
                    warnings.extend(linguistic_warnings)
                    unsupported.extend(candidates)
                chunks.append(transformed)
                applied_list.extend(segment_applied)
            spoken = "".join(chunks)
            applied = tuple(applied_list)
        return PronunciationResult(
            written_text=normalized,
            spoken_text=spoken,
            language=selected_profile.language,
            profile=selected_profile,
            applied_rules=applied,
            warnings=tuple(warnings),
            unsupported_fragments=tuple(
                dict.fromkeys(
                    unsupported
                    if selected_profile.enabled and manual_override is None
                    else []
                )
            ),
            source_hash=source_digest,
            rules_hash=digest,
            engine_version=self.version,
        )
