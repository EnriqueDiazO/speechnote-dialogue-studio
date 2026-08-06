from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from dialogue_studio.models import DialogueProject
from dialogue_studio.paths import AppPaths
from dialogue_studio.pronunciation import PronunciationProfile, PronunciationRule
from dialogue_studio.pronunciation.engine import rules_hash
from dialogue_studio.pronunciation.import_export import (
    GlobalPronunciationStore,
    PendingTerm,
    PendingTermStore,
    detect_rule_conflicts,
    export_rules_csv,
    export_rules_json,
    merge_imported_rules,
    preview_rule_import,
    record_rule_usage,
    update_rule_with_audit,
)
from dialogue_studio.service import (
    add_utterance,
    audio_input_fingerprint,
    effective_pronunciation_result,
    mark_global_pronunciation_change,
    persist_pronunciation_result,
    update_project_pronunciation_rules,
    update_pronunciation_profile,
    update_utterance_pronunciation,
)
from dialogue_studio.storage import ProjectStore, deterministic_json


def make_rule(
    pattern: str,
    replacement: str,
    *,
    scope: str = "global",
    kind: str = "literal",
) -> PronunciationRule:
    return PronunciationRule.create(
        scope=scope,  # type: ignore[arg-type]
        language="es",
        kind=kind,  # type: ignore[arg-type]
        pattern=pattern,
        replacement=replacement,
    )


def test_legacy_project_loads_defaults_without_being_rewritten(tmp_path: Path) -> None:
    store = ProjectStore(AppPaths(tmp_path / "Music"))
    project = DialogueProject.new()
    legacy = project.to_dict()
    legacy.pop("pronunciation_profile")
    legacy.pop("pronunciation_rules")
    for utterance in legacy["utterances"]:
        utterance.pop("written_text")
        for key in list(utterance):
            if key.startswith("pronunciation") or key in {
                "use_pronunciation_engine",
                "manual_spoken_text_override",
                "utterance_rules",
                "spoken_text",
                "written_text_hash",
                "spoken_text_hash",
                "applied_pronunciation_rule_ids",
            }:
                utterance.pop(key)
    directory = store.create_directory(project)
    project_file = directory / "project.json"
    original = deterministic_json(legacy)
    project_file.write_text(original, encoding="utf-8")

    loaded = store.load(directory)

    assert loaded.pronunciation_profile.enabled
    assert loaded.pronunciation_profile.language == "es"
    assert loaded.utterances[0].use_pronunciation_engine
    assert not loaded.pronunciation_rules
    assert project_file.read_text(encoding="utf-8") == original


def test_project_and_utterance_pronunciation_round_trip(tmp_path: Path) -> None:
    project = DialogueProject.new()
    utterance = project.utterances[0]
    project_rule = make_rule("Qwen", "cuen", scope="project")
    local_rule = make_rule("Haseman", "jásiman", scope="utterance")
    project.pronunciation_rules = [project_rule]
    project.pronunciation_profile = PronunciationProfile(language="es", math_style="explicit")
    utterance.text = "Qwen y Haseman"
    utterance.utterance_rules = [local_rule]
    utterance.manual_spoken_text_override = "lectura manual"
    result = effective_pronunciation_result(project, utterance)
    persist_pronunciation_result(utterance, result)

    store = ProjectStore(AppPaths(tmp_path / "Music"))
    directory = store.save(project)
    loaded = store.load(directory)
    saved = json.loads((directory / "project.json").read_text(encoding="utf-8"))

    assert loaded.to_dict() == project.to_dict()
    assert saved["utterances"][0]["written_text"] == "Qwen y Haseman"
    assert saved["utterances"][0]["spoken_text"] == "lectura manual"
    assert saved["pronunciation_profile"]["math_style"] == "explicit"
    assert loaded.utterances[0].utterance_rules[0].scope == "utterance"


def test_global_dictionary_is_atomic_and_corruption_preserves_original(
    tmp_path: Path,
) -> None:
    paths = AppPaths(tmp_path / "Music")
    store = GlobalPronunciationStore(paths)
    rule = make_rule("Qwen", "cuen")
    target = store.save([rule])
    assert store.load().rules == (rule,)
    target.write_text("{broken", encoding="utf-8")

    recovered = store.load()

    assert not recovered.rules
    assert recovered.warnings
    assert target.read_text(encoding="utf-8") == "{broken"
    assert recovered.recovery_copy is not None
    assert recovered.recovery_copy.read_text(encoding="utf-8") == "{broken"


def test_json_csv_import_preview_conflicts_and_incremental_modes() -> None:
    original = make_rule("Qwen", "cuen")
    shadow = make_rule("Qwen", "kuen")
    cycle = make_rule("cuen", "Qwen")
    invalid = original.to_dict() | {"rule_id": "invalid", "pattern": ""}
    payload = json.dumps(
        {"schema_version": 1, "rules": [shadow.to_dict(), cycle.to_dict(), invalid]}
    )

    preview = preview_rule_import(
        payload,
        format_name="json",
        scope="global",
        existing=[original],
    )

    assert len(preview.valid_rules) == 2
    assert len(preview.rejected_rules) == 1
    assert {item.kind for item in preview.conflicts} >= {"shadowing", "cycle"}
    disabled = merge_imported_rules([], preview.valid_rules, mode="disabled")
    assert all(not rule.enabled for rule in disabled)
    updated = merge_imported_rules(
        [original],
        [replace(original, replacement="kuen")],
        mode="update",
    )
    assert updated[0].replacement == "kuen"

    csv_preview = preview_rule_import(
        export_rules_csv([original]),
        format_name="csv",
        scope="global",
    )
    assert csv_preview.valid_rules == (original,)
    assert json.loads(export_rules_json([original]))["schema_version"] == 1


def test_regex_group_conflict_usage_and_audit_do_not_change_behavior_hash() -> None:
    regex = make_rule(r"(red)", r"\2", kind="regex")
    conflicts = detect_rule_conflicts(regex, [])
    assert conflicts[0].kind == "regex_group"

    edited = update_rule_with_audit(
        replace(regex, replacement=r"\1"),
        changes={"notes": "Revisada"},
        context="project:test",
    )
    used = record_rule_usage([edited], {edited.rule_id})[0]
    assert used.usage_count == 1
    assert used.change_history[-1]["context"] == "project:test"
    assert used.behavior_dict() == edited.behavior_dict()
    assert rules_hash([used]) == rules_hash([edited])

    corrupt = preview_rule_import("{broken", format_name="json", scope="global")
    assert corrupt.rejected_rules[0].message.startswith("JSON corrupto")


def test_pending_terms_require_confirmation_and_respect_ignore_always(tmp_path: Path) -> None:
    store = PendingTermStore(AppPaths(tmp_path / "Music"))
    first = PendingTerm.create(
        "MOFA2",
        language="es",
        context="Usaremos MOFA2 en el análisis",
        source="utterance",
    )
    terms = store.record(first)
    assert terms[0].status == "pending"
    assert terms[0].occurrences == 1
    terms = store.record(
        PendingTerm.create(
            "MOFA2", language="es", context="MOFA2", source="subtitle"
        )
    )
    assert terms[0].occurrences == 2
    store.set_status(first.candidate_id, "ignored_always")
    terms = store.record(
        PendingTerm.create("MOFA2", language="es", context="otra", source="paste")
    )
    assert terms[0].occurrences == 2
    assert terms[0].status == "ignored_always"


def test_rule_changes_mark_only_affected_audio_stale_without_deleting_files(
    tmp_path: Path,
) -> None:
    project = DialogueProject.new()
    first = project.utterances[0]
    first.text = "Qwen aprende"
    second = add_utterance(project, text="Texto ordinario")
    for utterance in (first, second):
        utterance.status = "ready"
        utterance.audio_relative_path = f"audio/{utterance.utterance_id}.wav"
        path = tmp_path / utterance.audio_relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old wav")

    affected = update_project_pronunciation_rules(
        project,
        [make_rule("Qwen", "cuen", scope="project")],
    )

    assert affected == [first.utterance_id]
    assert first.status == "stale"
    assert second.status == "ready"
    assert all((tmp_path / item.audio_relative_path).exists() for item in (first, second))

    new_global = [make_rule("ordinario", "especial")]
    affected = mark_global_pronunciation_change(project, old_rules=[], new_rules=new_global)
    assert affected == [second.utterance_id]
    assert second.status == "stale"


def test_profile_override_and_rules_participate_in_fingerprint() -> None:
    project = DialogueProject.new()
    utterance = project.utterances[0]
    utterance.text = "MSE = 0.05"
    baseline = audio_input_fingerprint(project, utterance)

    update_utterance_pronunciation(
        project,
        utterance.utterance_id,
        manual_override="eme ese e es igual a cero coma cero cinco",
    )
    manual = audio_input_fingerprint(project, utterance)
    update_utterance_pronunciation(project, utterance.utterance_id, manual_override=None)
    update_pronunciation_profile(
        project,
        replace(project.pronunciation_profile, math_style="explicit"),
    )
    profile = audio_input_fingerprint(project, utterance)

    assert len({baseline, manual, profile}) == 3
