"""Streamlit views for pronunciation dictionaries and spoken-text previews."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import streamlit as st

from ..models import DialogueProject, SpeakerProfile, Utterance
from ..paths import AppPaths
from ..service import (
    effective_pronunciation_result,
    mark_global_pronunciation_change,
    update_project_pronunciation_rules,
    update_pronunciation_profile,
    update_utterance_pronunciation,
)
from .corpus_export import build_corpus_candidate, export_corpus_candidate_json
from .engine import PronunciationEngine
from .glossary import builtin_rules
from .import_export import (
    GlobalPronunciationStore,
    ImportPreview,
    PendingTerm,
    PendingTermStore,
    detect_rule_conflicts,
    export_rules_csv,
    export_rules_json,
    merge_imported_rules,
    preview_rule_import,
    update_rule_with_audit,
)
from .models import PronunciationProfile, PronunciationResult, PronunciationRule


def global_rules() -> list[PronunciationRule]:
    return list(st.session_state.get("global_pronunciation_rules", []))


def _reset_audio_artifacts() -> None:
    st.session_state.master_wav = None
    st.session_state.master_mp3 = None
    st.session_state.project_zip = None


def _render_corpus_candidate_export(
    result: PronunciationResult,
    *,
    key_prefix: str,
    initial_category: str = "mixed_prose_math",
) -> None:
    st.caption(
        "Este archivo es un candidato. No se incorpora a las pruebas hasta ser "
        "revisado y promovido explícitamente."
    )
    fields = st.columns(2)
    case_id = fields[0].text_input(
        "case_id",
        value=(
            f"{result.language.split('-', 1)[0]}-"
            f"{initial_category.replace('_', '-')}-candidate-001"
        ),
        key=f"{key_prefix}-case-id",
    )
    category = fields[1].text_input(
        "Categoría del corpus",
        value=initial_category,
        key=f"{key_prefix}-category",
    )
    tags_text = st.text_input(
        "Tags separados por comas",
        key=f"{key_prefix}-tags",
    )
    notes = st.text_area(
        "Notas de revisión",
        key=f"{key_prefix}-notes",
        height=80,
    )
    try:
        candidate = build_corpus_candidate(
            result,
            case_id=case_id,
            category=category,
            tags=tags_text.split(","),
            notes=notes,
        )
        st.download_button(
            "Descargar candidato JSON",
            export_corpus_candidate_json(candidate),
            file_name=f"{candidate.case_id}.json",
            mime="application/json",
            key=f"{key_prefix}-download",
        )
    except ValueError as exc:
        st.error(str(exc))


def _save_rules(
    project: DialogueProject,
    paths: AppPaths,
    scope: str,
    rules: list[PronunciationRule],
) -> None:
    if scope == "global":
        previous = global_rules()
        GlobalPronunciationStore(paths).save(rules)
        st.session_state.global_pronunciation_rules = list(rules)
        mark_global_pronunciation_change(project, old_rules=previous, new_rules=rules)
    else:
        update_project_pronunciation_rules(
            project,
            rules,
            global_rules=global_rules(),
        )
    _reset_audio_artifacts()


def _new_rule_fields(prefix: str) -> None:
    columns = st.columns(2)
    columns[0].text_input("Patrón", key=f"{prefix}-pattern")
    columns[1].text_input("Pronunciación", key=f"{prefix}-replacement")
    options = st.columns(4)
    options[0].selectbox("Idioma", ["es", "en"], key=f"{prefix}-language")
    options[1].selectbox(
        "Tipo",
        ["literal", "phrase", "acronym", "math_alias", "regex"],
        key=f"{prefix}-kind",
    )
    options[2].text_input("Categoría", value="custom", key=f"{prefix}-category")
    options[3].number_input(
        "Prioridad",
        min_value=-10_000,
        max_value=10_000,
        value=0,
        key=f"{prefix}-priority",
    )
    flags = st.columns(2)
    flags[0].checkbox("Palabra completa", value=True, key=f"{prefix}-whole-word")
    flags[1].checkbox("Distinguir mayúsculas", key=f"{prefix}-case-sensitive")
    st.text_input("Notas", key=f"{prefix}-notes")
    if st.session_state.get(f"{prefix}-kind") == "regex":
        st.warning(
            "Opción avanzada: la expresión regular se validará antes de guardarse."
        )


def _rule_from_state(scope: str, prefix: str) -> PronunciationRule:
    return PronunciationRule.create(
        scope=scope,  # type: ignore[arg-type]
        language=st.session_state.get(f"{prefix}-language", "es"),
        kind=st.session_state.get(f"{prefix}-kind", "literal"),
        pattern=st.session_state.get(f"{prefix}-pattern", ""),
        replacement=st.session_state.get(f"{prefix}-replacement", ""),
        whole_word=st.session_state.get(f"{prefix}-whole-word", True),
        case_sensitive=st.session_state.get(f"{prefix}-case-sensitive", False),
        priority=int(st.session_state.get(f"{prefix}-priority", 0)),
        category=st.session_state.get(f"{prefix}-category", "custom"),
        notes=st.session_state.get(f"{prefix}-notes", ""),
    )


def _render_rule_collection(
    project: DialogueProject,
    paths: AppPaths,
    *,
    scope: str,
) -> None:
    rules = global_rules() if scope == "global" else list(project.pronunciation_rules)
    label = "global" if scope == "global" else "del proyecto"
    prefix = f"pronunciation-{scope}-new"
    st.markdown(f"#### Agregar regla {label}")
    _new_rule_fields(prefix)
    conflict_key = f"pronunciation-conflict-candidate-{scope}"
    if st.button("Agregar", key=f"pronunciation-add-{scope}"):
        try:
            candidate = _rule_from_state(scope, prefix)
            conflicts = detect_rule_conflicts(candidate, rules)
            if conflicts:
                st.session_state[conflict_key] = (candidate, conflicts)
            else:
                _save_rules(project, paths, scope, [*rules, candidate])
                st.toast("Regla agregada")
                st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    pending = st.session_state.get(conflict_key)
    if pending:
        candidate, conflicts = pending
        st.warning("La regla tiene conflictos y aún no se guardó.")
        for conflict in conflicts:
            st.caption(f"{conflict.kind}: {conflict.message}")
        actions = st.columns(3)
        if actions[0].button(
            "Guardar regla con conflictos",
            key=f"pronunciation-confirm-conflict-{scope}",
        ):
            _save_rules(project, paths, scope, [*rules, candidate])
            st.session_state.pop(conflict_key, None)
            st.rerun()
        if actions[1].button(
            "Fusionar con existente",
            key=f"pronunciation-merge-conflict-{scope}",
            disabled=not any(conflict.related_rule_id for conflict in conflicts),
        ):
            related_id = next(
                conflict.related_rule_id
                for conflict in conflicts
                if conflict.related_rule_id
            )
            existing = next(rule for rule in rules if rule.rule_id == related_id)
            merged = update_rule_with_audit(
                existing,
                changes={
                    "language": candidate.language,
                    "kind": candidate.kind,
                    "pattern": candidate.pattern,
                    "replacement": candidate.replacement,
                    "enabled": candidate.enabled,
                    "priority": candidate.priority,
                    "case_sensitive": candidate.case_sensitive,
                    "whole_word": candidate.whole_word,
                    "category": candidate.category,
                    "notes": candidate.notes,
                },
                context=f"{scope}:conflict-merge",
            )
            _save_rules(
                project,
                paths,
                scope,
                [merged if rule.rule_id == related_id else rule for rule in rules],
            )
            st.session_state.pop(conflict_key, None)
            st.rerun()
        if actions[2].button(
            "Descartar",
            key=f"pronunciation-discard-conflict-{scope}",
        ):
            st.session_state.pop(conflict_key, None)
            st.rerun()

    st.markdown(f"#### Reglas {label}")
    filters = st.columns(4)
    query = filters[0].text_input("Buscar", key=f"pronunciation-{scope}-search")
    language = filters[1].selectbox(
        "Idioma",
        ["todos", "es", "en"],
        key=f"pronunciation-{scope}-language-filter",
    )
    kind = filters[2].selectbox(
        "Tipo",
        ["todos", "literal", "phrase", "acronym", "math_alias", "regex"],
        key=f"pronunciation-{scope}-kind-filter",
    )
    state = filters[3].selectbox(
        "Estado",
        ["todas", "activas", "desactivadas"],
        key=f"pronunciation-{scope}-state-filter",
    )
    visible = [
        rule
        for rule in rules
        if (not query or query.casefold() in f"{rule.pattern} {rule.replacement}".casefold())
        and (language == "todos" or rule.language.startswith(language))
        and (kind == "todos" or rule.kind == kind)
        and (state == "todas" or rule.enabled == (state == "activas"))
    ]
    if not visible:
        st.caption("No hay reglas que coincidan con los filtros.")
    for rule in visible:
        with st.expander(f"{rule.pattern} → {rule.replacement}"):
            st.caption(
                f"{rule.language} · {rule.kind} · prioridad {rule.priority} · "
                f"usos {rule.usage_count} · id {rule.rule_id[:8]}"
            )
            enabled = st.checkbox(
                "Activa",
                value=rule.enabled,
                key=f"pronunciation-rule-enabled-{rule.rule_id}",
            )
            columns = st.columns(2)
            pattern = columns[0].text_input(
                "Patrón",
                value=rule.pattern,
                key=f"pronunciation-rule-pattern-{rule.rule_id}",
            )
            replacement_text = columns[1].text_input(
                "Pronunciación",
                value=rule.replacement,
                key=f"pronunciation-rule-replacement-{rule.rule_id}",
            )
            notes = st.text_input(
                "Notas",
                value=rule.notes,
                key=f"pronunciation-rule-notes-{rule.rule_id}",
            )
            properties = st.columns(4)
            language_value = properties[0].selectbox(
                "Idioma de la regla",
                ["es", "en"],
                index=0 if rule.language.startswith("es") else 1,
                key=f"pronunciation-rule-language-{rule.rule_id}",
            )
            kinds = ["literal", "phrase", "acronym", "math_alias", "regex"]
            kind_value = properties[1].selectbox(
                "Tipo de regla",
                kinds,
                index=kinds.index(rule.kind),
                key=f"pronunciation-rule-kind-{rule.rule_id}",
            )
            category = properties[2].text_input(
                "Categoría de la regla",
                value=rule.category,
                key=f"pronunciation-rule-category-{rule.rule_id}",
            )
            priority = properties[3].number_input(
                "Prioridad de la regla",
                min_value=-10_000,
                max_value=10_000,
                value=rule.priority,
                key=f"pronunciation-rule-priority-{rule.rule_id}",
            )
            match_flags = st.columns(2)
            whole_word = match_flags[0].checkbox(
                "Coincidencia de palabra completa",
                value=rule.whole_word,
                key=f"pronunciation-rule-whole-word-{rule.rule_id}",
            )
            case_sensitive = match_flags[1].checkbox(
                "Coincidencia sensible a mayúsculas",
                value=rule.case_sensitive,
                key=f"pronunciation-rule-case-sensitive-{rule.rule_id}",
            )
            if kind_value == "regex":
                st.warning("Regla regex avanzada: valida el patrón con Probar regla.")
            test_text = st.text_input(
                "Texto para probar la regla",
                key=f"pronunciation-rule-test-text-{rule.rule_id}",
            )
            if st.button("Probar regla", key=f"pronunciation-rule-test-{rule.rule_id}"):
                result = PronunciationEngine().transform(
                    test_text,
                    profile=PronunciationProfile(language=rule.language),
                    rules=[rule],
                )
                st.session_state[f"pronunciation-rule-result-{rule.rule_id}"] = (
                    result.spoken_text
                )
            tested = st.session_state.get(f"pronunciation-rule-result-{rule.rule_id}")
            if tested:
                st.code(tested, language=None)
            actions = st.columns(3)
            changed = (
                enabled != rule.enabled
                or pattern != rule.pattern
                or replacement_text != rule.replacement
                or notes != rule.notes
                or language_value != rule.language
                or kind_value != rule.kind
                or category != rule.category
                or priority != rule.priority
                or whole_word != rule.whole_word
                or case_sensitive != rule.case_sensitive
            )
            if actions[0].button(
                "Guardar cambios",
                key=f"pronunciation-rule-save-{rule.rule_id}",
                disabled=not changed,
            ):
                try:
                    updated = update_rule_with_audit(
                        rule,
                        changes={
                            "enabled": enabled,
                            "pattern": pattern,
                            "replacement": replacement_text,
                            "notes": notes,
                            "language": language_value,
                            "kind": kind_value,
                            "category": category,
                            "priority": int(priority),
                            "whole_word": whole_word,
                            "case_sensitive": case_sensitive,
                        },
                        context=f"{scope}:ui",
                    )
                    conflicts = detect_rule_conflicts(
                        updated,
                        [item for item in rules if item.rule_id != rule.rule_id],
                    )
                    if conflicts:
                        st.error("No se guardó: resuelve primero los conflictos.")
                    else:
                        _save_rules(
                            project,
                            paths,
                            scope,
                            [updated if item.rule_id == rule.rule_id else item for item in rules],
                        )
                        st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            if actions[1].button(
                "Duplicar",
                key=f"pronunciation-rule-duplicate-{rule.rule_id}",
            ):
                duplicate = PronunciationRule.create(
                    scope=scope,  # type: ignore[arg-type]
                    language=rule.language,
                    kind=rule.kind,
                    pattern=rule.pattern,
                    replacement=rule.replacement,
                    priority=rule.priority,
                    case_sensitive=rule.case_sensitive,
                    whole_word=rule.whole_word,
                    category=rule.category,
                    notes=f"Copia de {rule.rule_id[:8]}",
                    enabled=False,
                )
                _save_rules(project, paths, scope, [*rules, duplicate])
                st.rerun()
            if actions[2].button(
                "Eliminar",
                key=f"pronunciation-rule-delete-{rule.rule_id}",
            ):
                _save_rules(
                    project,
                    paths,
                    scope,
                    [item for item in rules if item.rule_id != rule.rule_id],
                )
                st.rerun()
            if rule.change_history:
                history = " · ".join(
                    f"{event.get('changed_at', '')} ({event.get('context', '')})"
                    for event in rule.change_history[-3:]
                )
                st.caption(f"Historial: {history}")


def _render_preview(
    project: DialogueProject,
    paths: AppPaths,
    capabilities: Any,
    voice_preview: Callable[[SpeakerProfile, str], None],
) -> None:
    written = st.text_area(
        "Texto escrito",
        key="pronunciation-preview-written",
        height=130,
        placeholder=r"Ejemplo: $\frac{\partial L}{\partial x}$",
    )
    selectors = st.columns(2)
    language = selectors[0].selectbox(
        "Idioma",
        ["es", "en"],
        index=0 if project.pronunciation_profile.language.startswith("es") else 1,
        key="pronunciation-preview-language",
    )
    math_style = selectors[1].selectbox(
        "Perfil matemático",
        ["concise", "classroom", "explicit", "symbolic"],
        index=["concise", "classroom", "explicit", "symbolic"].index(
            project.pronunciation_profile.math_style
        ),
        key="pronunciation-preview-math-style",
    )
    policies = st.columns(2)
    acronym = policies[0].selectbox(
        "Política de siglas",
        ["custom", "spell_out", "read_as_word", "preserve"],
        key="pronunciation-preview-acronym-policy",
    )
    number = policies[1].selectbox(
        "Lectura de números",
        ["natural", "digits", "preserve"],
        key="pronunciation-preview-number-style",
    )
    if st.button("Transformar", key="pronunciation-preview-transform", type="primary"):
        profile = PronunciationProfile(
            language=language,
            math_style=math_style,  # type: ignore[arg-type]
            acronym_policy=acronym,  # type: ignore[arg-type]
            number_style=number,  # type: ignore[arg-type]
            unit_style=project.pronunciation_profile.unit_style,
            punctuation_style=project.pronunciation_profile.punctuation_style,
        )
        st.session_state.pronunciation_preview_result = PronunciationEngine().transform(
            written,
            profile=profile,
            rules=[*global_rules(), *project.pronunciation_rules],
        )
    result = st.session_state.get("pronunciation_preview_result")
    if result is not None:
        st.markdown("#### Texto hablado resultante")
        st.code(result.spoken_text, language=None)
        actions = st.columns(3)
        if actions[0].button("Copiar texto hablado"):
            st.toast("Usa el icono de copiar del bloque de texto hablado.")
        speaker_ids = [speaker.speaker_id for speaker in project.speakers]
        selected = actions[1].selectbox(
            "Voz seleccionada",
            speaker_ids,
            format_func=lambda value: project.speaker(value).name,
            key="pronunciation-preview-speaker",
            label_visibility="collapsed",
        )
        speaker = project.speaker(selected)
        provider_available = (
            capabilities.qwen_available
            if speaker.tts_config.provider == "qwen"
            else capabilities.speechnote_available
        )
        if actions[2].button(
            "Probar con voz seleccionada",
            disabled=(
                not provider_available
                or capabilities.has_active_synthesis
                or not speaker.tts_config.voice_id
            ),
        ):
            try:
                voice_preview(speaker, result.spoken_text)
                st.rerun()
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(str(exc))
        preview_audio = st.session_state.get("pronunciation_preview_audio")
        if preview_audio and Path(preview_audio).is_file():
            st.audio(preview_audio, format="audio/wav")
        if result.applied_rules:
            st.dataframe(
                [asdict(item) for item in result.applied_rules],
                use_container_width=True,
                hide_index=True,
            )
        for warning in result.warnings:
            st.warning(warning.message)
        if result.unsupported_fragments:
            st.caption("Fragmentos no soportados: " + ", ".join(result.unsupported_fragments))
        with st.expander("Exportar como caso de corpus"):
            _render_corpus_candidate_export(
                result,
                key_prefix="pronunciation-preview-corpus",
            )
        quick = st.columns(4)
        quick[0].text_input("Término", key="pronunciation-preview-quick-term")
        quick[1].text_input("Pronunciación", key="pronunciation-preview-quick-spoken")
        for column, scope, label in (
            (quick[2], "project", "Agregar al proyecto"),
            (quick[3], "global", "Agregar global"),
        ):
            if not column.button(label, key=f"pronunciation-preview-quick-add-{scope}"):
                continue
            try:
                rule = PronunciationRule.create(
                    scope=scope,  # type: ignore[arg-type]
                    language=result.language,
                    kind="literal",
                    pattern=st.session_state.get("pronunciation-preview-quick-term", ""),
                    replacement=st.session_state.get(
                        "pronunciation-preview-quick-spoken", ""
                    ),
                )
                current = global_rules() if scope == "global" else project.pronunciation_rules
                if detect_rule_conflicts(rule, current):
                    st.error("La regla entra en conflicto con otra regla existente.")
                else:
                    _save_rules(project, paths, scope, [*current, rule])
                    st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if st.button("Restablecer", key="pronunciation-preview-reset"):
        st.session_state.pronunciation_preview_result = None
        st.session_state.pop("pronunciation_preview_audio", None)
        st.rerun()


def _render_builtins(project: DialogueProject) -> None:
    query = st.text_input("Buscar reglas incorporadas", key="pronunciation-builtin-search")
    language = st.selectbox(
        "Idioma de reglas incorporadas",
        ["es", "en"],
        index=0 if project.pronunciation_profile.language.startswith("es") else 1,
        key="pronunciation-builtin-language",
    )
    rules = [
        rule
        for rule in builtin_rules(language)
        if not query
        or query.casefold()
        in f"{rule.pattern} {rule.replacement} {rule.category}".casefold()
    ]
    st.caption(
        "Son de sólo lectura. Una regla global o de proyecto puede sobrescribirlas."
    )
    st.dataframe(
        [
            {
                "patrón": rule.pattern,
                "pronunciación": rule.replacement,
                "tipo": rule.kind,
                "categoría": rule.category,
            }
            for rule in rules
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_settings(project: DialogueProject) -> None:
    profile = project.pronunciation_profile
    enabled = st.checkbox(
        "Activar motor de pronunciación",
        value=profile.enabled,
        key=f"pronunciation-project-enabled-{project.project_id}",
    )
    columns = st.columns(5)
    math_style = columns[0].selectbox(
        "Lectura matemática",
        ["concise", "classroom", "explicit", "symbolic"],
        index=["concise", "classroom", "explicit", "symbolic"].index(profile.math_style),
        key=f"pronunciation-project-math-{project.project_id}",
    )
    acronym = columns[1].selectbox(
        "Siglas",
        ["custom", "spell_out", "read_as_word", "preserve"],
        index=["custom", "spell_out", "read_as_word", "preserve"].index(
            profile.acronym_policy
        ),
        key=f"pronunciation-project-acronym-{project.project_id}",
    )
    number = columns[2].selectbox(
        "Números",
        ["natural", "digits", "preserve"],
        index=["natural", "digits", "preserve"].index(profile.number_style),
        key=f"pronunciation-project-number-{project.project_id}",
    )
    units = columns[3].selectbox(
        "Unidades",
        ["natural", "spell_out", "preserve"],
        index=["natural", "spell_out", "preserve"].index(profile.unit_style),
        key=f"pronunciation-project-unit-{project.project_id}",
    )
    punctuation = columns[4].selectbox(
        "Puntuación",
        ["natural", "explicit", "preserve"],
        index=["natural", "explicit", "preserve"].index(profile.punctuation_style),
        key=f"pronunciation-project-punctuation-{project.project_id}",
    )
    desired = replace(
        profile,
        enabled=enabled,
        math_style=math_style,
        acronym_policy=acronym,
        number_style=number,
        unit_style=units,
        punctuation_style=punctuation,
    )
    if desired != profile:
        update_pronunciation_profile(project, desired)
        _reset_audio_artifacts()


def _render_import_export(project: DialogueProject, paths: AppPaths) -> None:
    scope = st.selectbox(
        "Alcance de importación",
        ["global", "project"],
        format_func=lambda value: "Global" if value == "global" else "Proyecto",
        key="pronunciation-import-scope",
    )
    rules = global_rules() if scope == "global" else list(project.pronunciation_rules)
    downloads = st.columns(2)
    downloads[0].download_button(
        "Exportar JSON",
        export_rules_json(rules),
        file_name=f"pronunciation-{scope}.json",
        mime="application/json",
    )
    downloads[1].download_button(
        "Exportar CSV",
        export_rules_csv(rules),
        file_name=f"pronunciation-{scope}.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader(
        "Importar JSON o CSV",
        type=["json", "csv"],
        key="pronunciation-import-file",
    )
    if uploaded is not None and st.button("Validar importación"):
        format_name = "csv" if uploaded.name.lower().endswith(".csv") else "json"
        st.session_state.pronunciation_import_preview = preview_rule_import(
            uploaded.getvalue().decode("utf-8"),
            format_name=format_name,
            scope=scope,  # type: ignore[arg-type]
            existing=rules,
        )
    preview: ImportPreview | None = st.session_state.get("pronunciation_import_preview")
    if preview is None:
        return
    metrics = st.columns(3)
    metrics[0].metric("Válidas", len(preview.valid_rules))
    metrics[1].metric("Rechazadas", len(preview.rejected_rules))
    metrics[2].metric("Conflictos", len(preview.conflicts))
    for rejected in preview.rejected_rules:
        st.error(f"Fila {rejected.index}: {rejected.message}")
    for conflict in preview.conflicts:
        st.warning(f"{conflict.kind}: {conflict.message}")
    conflict_ids = {conflict.rule_id for conflict in preview.conflicts}
    selected_rule_ids = st.multiselect(
        "Reglas válidas que se aplicarán",
        [rule.rule_id for rule in preview.valid_rules],
        default=[
            rule.rule_id for rule in preview.valid_rules if rule.rule_id not in conflict_ids
        ],
        format_func=lambda value: next(
            f"{rule.pattern} → {rule.replacement}"
            for rule in preview.valid_rules
            if rule.rule_id == value
        ),
        key="pronunciation-import-selected-rules",
    )
    mode = st.selectbox(
        "Modo incremental",
        ["add", "update", "disabled"],
        format_func={
            "add": "Agregar sólo nuevas",
            "update": "Actualizar por rule_id",
            "disabled": "Importar desactivadas para revisar",
        }.get,
        key="pronunciation-import-mode",
    )
    st.caption("La importación nunca reemplaza el diccionario completo.")
    if st.button("Aplicar importación incremental", disabled=not selected_rule_ids):
        selected_rules = [
            rule for rule in preview.valid_rules if rule.rule_id in selected_rule_ids
        ]
        merged = list(merge_imported_rules(rules, selected_rules, mode=mode))
        _save_rules(project, paths, scope, merged)
        st.session_state.pronunciation_import_preview = None
        st.rerun()


def _render_pending_terms(project: DialogueProject, paths: AppPaths) -> None:
    terms = list(st.session_state.get("pronunciation_pending_terms", []))
    pending = [term for term in terms if term.status in {"pending", "postponed"}]
    st.caption("Los candidatos nunca se convierten en reglas sin confirmación.")
    if not pending:
        st.info("No hay términos por revisar.")
        return
    store = PendingTermStore(paths)
    for term in pending:
        with st.container(border=True, key=f"pending-term-{term.candidate_id}"):
            st.markdown(f"**{term.term}** · {term.language} · {term.occurrences} usos")
            st.caption(term.context or "Sin contexto")
            pronunciation = st.text_input(
                "Pronunciación confirmada",
                key=f"pending-pronunciation-{term.candidate_id}",
            )
            scope = st.selectbox(
                "Guardar en",
                ["project", "global"],
                format_func=lambda value: "Proyecto" if value == "project" else "Global",
                key=f"pending-scope-{term.candidate_id}",
            )
            actions = st.columns(4)
            if actions[0].button(
                "Crear regla",
                key=f"pending-add-{term.candidate_id}",
                disabled=not pronunciation.strip(),
            ):
                rule = PronunciationRule.create(
                    scope=scope,  # type: ignore[arg-type]
                    language=term.language,
                    kind="acronym" if term.term.isupper() else "literal",
                    pattern=term.term,
                    replacement=pronunciation,
                    category=term.category,
                    notes=f"Confirmada desde {term.source}",
                )
                current = global_rules() if scope == "global" else project.pronunciation_rules
                if detect_rule_conflicts(rule, current):
                    st.error("Existe una regla conflictiva; revísala en el editor.")
                else:
                    _save_rules(project, paths, scope, [*current, rule])
                    st.session_state.pronunciation_pending_terms = list(
                        store.set_status(term.candidate_id, "added")
                    )
                    st.rerun()
            for column, label, status in (
                (actions[1], "Ignorar", "ignored"),
                (actions[2], "Ignorar siempre", "ignored_always"),
                (actions[3], "Posponer", "postponed"),
            ):
                if column.button(label, key=f"pending-{status}-{term.candidate_id}"):
                    st.session_state.pronunciation_pending_terms = list(
                        store.set_status(term.candidate_id, status)  # type: ignore[arg-type]
                    )
                    st.rerun()


def render_pronunciation(
    project: DialogueProject,
    paths: AppPaths,
    capabilities: Any,
    voice_preview: Callable[[SpeakerProfile, str], None],
) -> None:
    st.markdown("## Pronunciación")
    st.caption(
        "El texto escrito se conserva intacto. Esta capa crea el mismo texto hablado "
        "para cualquier proveedor TTS."
    )
    for warning in st.session_state.get("pronunciation_store_warnings", []):
        st.warning(warning)
    tabs = st.tabs(
        [
            "Vista previa",
            "Diccionario global",
            "Diccionario del proyecto",
            "Reglas incorporadas",
            "Configuración matemática",
            "Importar y exportar",
            "Términos por revisar",
        ]
    )
    with tabs[0]:
        _render_preview(project, paths, capabilities, voice_preview)
    with tabs[1]:
        _render_rule_collection(project, paths, scope="global")
    with tabs[2]:
        _render_rule_collection(project, paths, scope="project")
    with tabs[3]:
        _render_builtins(project)
    with tabs[4]:
        _render_settings(project)
    with tabs[5]:
        _render_import_export(project, paths)
    with tabs[6]:
        _render_pending_terms(project, paths)


def _queue_pending_candidates(
    project: DialogueProject,
    paths: AppPaths,
    utterance: Utterance,
    fragments: tuple[str, ...],
) -> None:
    store = PendingTermStore(paths)
    current = tuple(st.session_state.get("pronunciation_pending_terms", []))
    for fragment in fragments:
        candidate = PendingTerm.create(
            fragment,
            language=project.pronunciation_profile.language,
            context=utterance.text,
            source="utterance_warning",
            project_id=project.project_id,
            utterance_id=utterance.utterance_id,
        )
        current = store.record(candidate)
    st.session_state.pronunciation_pending_terms = list(current)


def render_utterance_pronunciation(
    project: DialogueProject,
    utterance: Utterance,
    paths: AppPaths,
    *,
    disabled: bool,
) -> None:
    result = effective_pronunciation_result(
        project,
        utterance,
        global_rules=global_rules(),
    )
    status = "desactualizado" if utterance.status == "stale" else "actualizado"
    mode = "override manual" if utterance.manual_spoken_text_override else "automático"
    st.caption(f"Texto para voz: {mode} · Estado: {status}")
    automatic = st.checkbox(
        "Usar motor automático",
        value=utterance.use_pronunciation_engine,
        key=f"utterance-pronunciation-enabled-{utterance.utterance_id}",
        disabled=disabled,
    )
    st.checkbox(
        "Si falla la transformación, permitir usar explícitamente el texto escrito",
        value=False,
        key=f"utterance-pronunciation-fallback-{utterance.utterance_id}",
        disabled=disabled,
    )
    manual_enabled = st.checkbox(
        "Editar pronunciación sólo aquí",
        value=utterance.manual_spoken_text_override is not None,
        key=f"utterance-pronunciation-manual-{utterance.utterance_id}",
        disabled=disabled,
    )
    manual = ""
    if manual_enabled:
        manual = st.text_area(
            "Texto hablado manual",
            value=utterance.manual_spoken_text_override or result.spoken_text,
            key=f"utterance-pronunciation-manual-text-{utterance.utterance_id}",
            disabled=disabled,
        )
    if automatic != utterance.use_pronunciation_engine:
        update_utterance_pronunciation(
            project,
            utterance.utterance_id,
            enabled=automatic,
            global_rules=global_rules(),
        )
        _reset_audio_artifacts()
        st.rerun()
    actions = st.columns(3)
    if actions[0].button(
        "Aplicar pronunciación",
        key=f"utterance-pronunciation-apply-{utterance.utterance_id}",
        disabled=disabled or not manual_enabled,
    ):
        update_utterance_pronunciation(
            project,
            utterance.utterance_id,
            manual_override=manual,
            global_rules=global_rules(),
        )
        _reset_audio_artifacts()
        st.rerun()
    if actions[1].button(
        "Restablecer override",
        key=f"utterance-pronunciation-reset-{utterance.utterance_id}",
        disabled=disabled or utterance.manual_spoken_text_override is None,
    ):
        update_utterance_pronunciation(
            project,
            utterance.utterance_id,
            manual_override=None,
            global_rules=global_rules(),
        )
        _reset_audio_artifacts()
        st.rerun()
    trace_key = f"utterance-pronunciation-trace-{utterance.utterance_id}"
    if actions[2].button("Ver transformación", key=trace_key):
        st.session_state[f"{trace_key}-visible"] = not st.session_state.get(
            f"{trace_key}-visible", False
        )
    st.markdown("**Vista previa del texto hablado**")
    st.code(result.spoken_text, language=None)
    for warning in result.warnings:
        st.warning(warning.message)
    show_corpus_export = st.checkbox(
        "Exportar lectura como caso candidato",
        key=f"utterance-pronunciation-corpus-visible-{utterance.utterance_id}",
    )
    if show_corpus_export:
        with st.container(border=True):
            _render_corpus_candidate_export(
                result,
                key_prefix=f"utterance-pronunciation-corpus-{utterance.utterance_id}",
            )
    if result.unsupported_fragments and st.button(
        "Enviar términos por revisar",
        key=f"utterance-pronunciation-pending-{utterance.utterance_id}",
    ):
        _queue_pending_candidates(
            project,
            paths,
            utterance,
            result.unsupported_fragments,
        )
        st.toast("Candidatos añadidos; aún no son reglas.")
    if st.session_state.get(f"{trace_key}-visible"):
        st.dataframe(
            [asdict(item) for item in result.applied_rules],
            use_container_width=True,
            hide_index=True,
        )
    quick = st.columns(4)
    term = quick[0].text_input(
        "Término",
        key=f"utterance-pronunciation-term-{utterance.utterance_id}",
    )
    spoken = quick[1].text_input(
        "Pronunciación",
        key=f"utterance-pronunciation-spoken-{utterance.utterance_id}",
    )
    if quick[2].button(
        "Agregar regla sólo aquí",
        key=f"utterance-pronunciation-add-local-{utterance.utterance_id}",
        disabled=disabled,
    ):
        try:
            rule = PronunciationRule.create(
                scope="utterance",
                language=project.pronunciation_profile.language,
                kind="literal",
                pattern=term,
                replacement=spoken,
            )
            update_utterance_pronunciation(
                project,
                utterance.utterance_id,
                rules=[*utterance.utterance_rules, rule],
                global_rules=global_rules(),
            )
            _reset_audio_artifacts()
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if quick[3].button(
        "Agregar término al diccionario del proyecto",
        key=f"utterance-pronunciation-add-project-{utterance.utterance_id}",
        disabled=disabled,
    ):
        try:
            rule = PronunciationRule.create(
                scope="project",
                language=project.pronunciation_profile.language,
                kind="literal",
                pattern=term,
                replacement=spoken,
            )
            if detect_rule_conflicts(rule, project.pronunciation_rules):
                st.error("El término entra en conflicto con otra regla del proyecto.")
            else:
                _save_rules(
                    project,
                    paths,
                    "project",
                    [*project.pronunciation_rules, rule],
                )
                st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if utterance.utterance_rules:
        st.caption("Reglas locales: " + ", ".join(
            f"{rule.pattern} → {rule.replacement}" for rule in utterance.utterance_rules
        ))
        promote = st.selectbox(
            "Promover regla local",
            [rule.rule_id for rule in utterance.utterance_rules],
            format_func=lambda value: next(
                rule.pattern for rule in utterance.utterance_rules if rule.rule_id == value
            ),
            key=f"utterance-pronunciation-promote-choice-{utterance.utterance_id}",
        )
        promotion_actions = st.columns(2)
        for column, scope, label in (
            (promotion_actions[0], "project", "Promover al proyecto"),
            (promotion_actions[1], "global", "Promover a global"),
        ):
            if not column.button(
                label,
                key=f"utterance-pronunciation-promote-{scope}-{utterance.utterance_id}",
            ):
                continue
            source = next(rule for rule in utterance.utterance_rules if rule.rule_id == promote)
            promoted = PronunciationRule.create(
                scope=scope,  # type: ignore[arg-type]
                language=source.language,
                kind=source.kind,
                pattern=source.pattern,
                replacement=source.replacement,
                priority=source.priority,
                case_sensitive=source.case_sensitive,
                whole_word=source.whole_word,
                category=source.category,
                notes=f"Promovida desde intervención {utterance.order}",
            )
            current = global_rules() if scope == "global" else project.pronunciation_rules
            _save_rules(project, paths, scope, [*current, promoted])
            st.rerun()
