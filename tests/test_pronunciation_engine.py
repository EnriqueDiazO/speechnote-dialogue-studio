from __future__ import annotations

import unicodedata

import pytest

from dialogue_studio.pronunciation import (
    PronunciationEngine,
    PronunciationProfile,
    PronunciationRule,
)


def rule(
    pattern: str,
    replacement: str,
    *,
    scope: str = "project",
    kind: str = "literal",
    priority: int = 0,
    case_sensitive: bool = False,
    whole_word: bool = True,
    enabled: bool = True,
) -> PronunciationRule:
    return PronunciationRule.create(
        scope=scope,  # type: ignore[arg-type]
        language="es",
        kind=kind,  # type: ignore[arg-type]
        pattern=pattern,
        replacement=replacement,
        priority=priority,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
        enabled=enabled,
    )


def test_precedence_priority_longest_phrase_and_stable_trace() -> None:
    engine = PronunciationEngine()
    rules = [
        rule("red", "builtin", scope="builtin", priority=100),
        rule("red", "global", scope="global", priority=100),
        rule("red", "proyecto", scope="project", priority=1),
        rule("red neuronal", "red neuronal artificial", scope="project", priority=1),
        rule("red", "intervención", scope="utterance", priority=-100),
    ]
    result = engine.transform("red neuronal y red", rules=rules)
    assert result.spoken_text == "intervención neuronal y intervención"
    assert [item.scope for item in result.applied_rules] == ["utterance", "utterance"]
    assert result.applied_rules[0].source_span == (0, 3)

    only_project = engine.transform("red neuronal", rules=rules[:-1])
    assert only_project.spoken_text == "red neuronal artificial"


def test_whole_word_case_unicode_disabled_and_non_recursive() -> None:
    engine = PronunciationEngine()
    decomposed = unicodedata.normalize("NFD", "México")
    rules = [
        rule("pi", "pee"),
        rule("QWEN", "cuen", case_sensitive=True),
        rule("México", "méjico"),
        rule("uno", "dos"),
        rule("dos", "tres"),
        rule("oculto", "visible", enabled=False),
    ]
    result = engine.transform(f"pipeline pi Qwen QWEN {decomposed} uno oculto", rules=rules)
    assert result.spoken_text == "pipeline pee Qwen cuen méjico dos oculto"
    assert "tres" not in result.spoken_text


def test_regex_validation_cycles_manual_override_and_hashes() -> None:
    with pytest.raises(ValueError, match="regular inválida"):
        rule("(", "x", kind="regex")
    with pytest.raises(ValueError, match="cuantificadores"):
        rule("(a+)+$", "x", kind="regex")

    a = rule("a", "b")
    b = rule("b", "a")
    result = PronunciationEngine().transform("a b", rules=[a, b])
    assert result.spoken_text == "a b"
    assert result.warnings[0].code == "rule_cycle"
    assert len(result.source_hash) == len(result.rules_hash) == 64

    override = PronunciationEngine().transform(
        "texto original",
        rules=[rule("texto", "otra cosa")],
        manual_override="lectura elegida",
    )
    assert override.written_text == "texto original"
    assert override.spoken_text == "lectura elegida"
    assert override.applied_rules[0].rule_id == "manual-utterance-override"


def test_profile_serialization_and_disabled_engine_preserve_written_text() -> None:
    profile = PronunciationProfile.from_dict(
        {
            "enabled": False,
            "language": "en-US",
            "math_style": "explicit",
            "acronym_policy": "spell_out",
            "number_style": "digits",
            "unit_style": "spell_out",
            "punctuation_style": "explicit",
        }
    )
    result = PronunciationEngine().transform(
        "MSE",
        profile=profile,
        rules=[rule("MSE", "eme ese e")],
    )
    assert result.spoken_text == "MSE"
    assert profile.to_dict()["math_style"] == "explicit"
