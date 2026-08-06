from __future__ import annotations

import json

import pytest

from dialogue_studio.pronunciation import PronunciationEngine, PronunciationProfile
from dialogue_studio.pronunciation.glossary import builtin_rules
from dialogue_studio.pronunciation.segmentation import segment_text


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ("prosa simple", ["prose"]),
        (
            "Antes $x_i$ y $$y=2$$ después",
            ["prose", "math_inline", "prose", "math_display", "prose"],
        ),
        ("\\(x+1\\) y \\[y=2\\]", ["math_inline", "prose", "math_display"]),
        ("`x=1` y ```py\ny=2\n```", ["code", "prose", "code"]),
        ("https://example.test/x?y=2 y $x=1$", ["url", "prose", "math_inline"]),
        ("a@b.example /tmp/model/file.pt", ["email", "prose", "path"]),
        ("θ_{t+1}=θ_t-η∇L(θ_t).", ["math_inline", "prose"]),
        ("Texto **Markdown** y 😊", ["prose"]),
        ("", []),
    ],
)
def test_segmentation_preserves_text_positions(text: str, kinds: list[str]) -> None:
    segments = segment_text(text)
    assert [segment.kind for segment in segments] == kinds
    assert "".join(segment.text for segment in segments) == text
    assert all(text[segment.start : segment.end] == segment.text for segment in segments)


def test_multiline_formula_and_protected_content() -> None:
    text = "$$\\sum_{i=1}^{n}\nx_i$$ correo@example.org `MSE` MSE"
    segments = segment_text(text)
    assert segments[0].kind == "math_display"
    assert "\n" in segments[0].text
    result = PronunciationEngine().transform(text)
    assert "`MSE`" in result.spoken_text
    assert result.spoken_text.endswith("eme ese e")


def test_only_explicit_utterance_rule_can_change_url_or_code() -> None:
    from dialogue_studio.pronunciation import PronunciationRule

    global_rule = PronunciationRule.create(
        scope="global",
        language="es",
        kind="literal",
        pattern="API",
        replacement="a pe i",
    )
    utterance_rule = PronunciationRule.create(
        scope="utterance",
        language="es",
        kind="literal",
        pattern="API",
        replacement="api local",
    )
    text = "API https://example.test/API `API`"
    global_result = PronunciationEngine().transform(text, rules=[global_rule])
    assert global_result.spoken_text == "a pe i https://example.test/API `API`"
    local_result = PronunciationEngine().transform(
        text, rules=[global_rule, utterance_rule]
    )
    assert local_result.spoken_text == (
        "api local https://example.test/api local `api local`"
    )


def test_spanish_and_english_builtin_scientific_glossaries() -> None:
    spanish = PronunciationEngine().transform("MSE ReLU CNN learning rate")
    assert spanish.spoken_text == "eme ese e re lu ce ene ene tasa de aprendizaje"
    assert all(item.scope == "builtin" for item in spanish.applied_rules)

    english = PronunciationEngine().transform(
        "MSE ReLU LSTM p-value",
        profile=PronunciationProfile.for_language("en-US"),
    )
    assert english.spoken_text == "em ess ee ree loo el ess tee em p value"


def test_builtin_boundaries_do_not_corrupt_words() -> None:
    result = PronunciationEngine().transform("pipeline sinopsis pi lossless loss")
    assert result.spoken_text == "pipeline sinopsis pi lossless pérdida"


def test_builtin_resources_expand_with_another_validated_file(tmp_path) -> None:
    language_root = tmp_path / "es"
    language_root.mkdir(parents=True)
    (language_root / "computing.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "language": "es",
                "category": "computing",
                "rules": [
                    {
                        "kind": "literal",
                        "pattern": "CUDA",
                        "replacement": "cuda",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = builtin_rules("es", resource_root=tmp_path)
    assert [(item.pattern, item.replacement, item.category) for item in loaded] == [
        ("CUDA", "cuda", "computing")
    ]
