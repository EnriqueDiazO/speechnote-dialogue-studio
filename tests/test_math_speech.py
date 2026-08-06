from __future__ import annotations

import pytest

from dialogue_studio.pronunciation import PronunciationEngine, PronunciationProfile


@pytest.mark.parametrize(
    ("expression", "spanish_parts", "english_parts"),
    [
        (
            "$θ_{t+1}=θ_t-η∇L(θ_t)$",
            ("theta sub te más uno", "eta gradiente", "ele de theta sub te"),
            ("theta sub t plus one", "eta gradient", "L of theta sub t"),
        ),
        (
            r"$y = \sigma(Wx+b)$",
            ("ye igual a sigma", "doble u equis más be"),
            ("y equals sigma", "W x plus b"),
        ),
        (
            r"$\frac{\partial L}{\partial w_{ij}}$",
            ("derivada parcial de ele respecto de", "doble u sub i jota"),
            ("partial derivative of L with respect to", "w sub i j"),
        ),
        (
            r"$\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$",
            ("sumatoria desde i igual a uno", "ye estimada sub i al cuadrado"),
            ("sum from i equals one", "y hat sub i squared"),
        ),
        (
            r"$\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}$",
            ("soft max de zeta sub i", "dividido entre sumatoria", "zeta sub jota"),
            ("soft max of z sub i", "divided by sum", "z sub j"),
        ),
        (
            r"$\lim_{n\to\infty}\frac{1}{n}\sum_{i=1}^{n}X_i$",
            ("límite cuando ene tiende a infinito", "uno dividido entre ene", "equis sub i"),
            ("limit as n approaches infinity", "one divided by n", "X sub i"),
        ),
        (
            r"$\int_a^b f(x)\,dx$",
            ("integral desde a hasta be", "efe de equis", "diferencial de equis"),
            ("integral from a to b", "f of x", "d x"),
        ),
    ],
)
def test_required_equations_are_semantically_verbalized(
    expression: str,
    spanish_parts: tuple[str, ...],
    english_parts: tuple[str, ...],
) -> None:
    engine = PronunciationEngine()
    spanish = engine.transform(expression)
    english = engine.transform(
        expression,
        profile=PronunciationProfile.for_language("en"),
    )
    assert all(part in spanish.spoken_text for part in spanish_parts)
    assert all(part in english.spoken_text for part in english_parts)
    assert not spanish.warnings
    assert not english.warnings


@pytest.mark.parametrize(
    ("style", "spanish", "english"),
    [
        ("concise", "a sobre be", "a over b"),
        ("classroom", "a dividido entre be", "a divided by b"),
        (
            "explicit",
            "fracción con numerador a y denominador be",
            "fraction with numerator a and denominator b",
        ),
        ("symbolic", "fracción a, be", "fraction a, b"),
    ],
)
def test_fraction_math_profiles(style: str, spanish: str, english: str) -> None:
    engine = PronunciationEngine()
    es_profile = PronunciationProfile(language="es", math_style=style)  # type: ignore[arg-type]
    en_profile = PronunciationProfile(language="en", math_style=style)  # type: ignore[arg-type]
    assert engine.transform(r"$\frac{a}{b}$", profile=es_profile).spoken_text == spanish
    assert engine.transform(r"$\frac{a}{b}$", profile=en_profile).spoken_text == english


def test_roots_spaces_vectors_matrices_and_unknown_recovery() -> None:
    engine = PronunciationEngine()
    supported = engine.transform(
        r"$\sqrt[n]{x}; \|x\|; \langle x,y\rangle; "
        r"\mathbb{R}^n; A^T; A^{-1}; \begin{matrix}a&b\\c&d\end{matrix}$"
    )
    for expected in (
        "raíz de índice ene de equis",
        "norma de equis",
        "producto interno",
        "números reales elevado a ene",
        "a transpuesta",
        "a inversa",
        "matriz fila a columna be; fila ce columna de",
    ):
        assert expected in supported.spoken_text
    assert not supported.warnings

    unknown = engine.transform(r"$\unknowncommand{x}$")
    assert "unknowncommand equis" in unknown.spoken_text
    assert unknown.warnings[0].code == "unsupported_math_command"
    assert unknown.unsupported_fragments == (r"\unknowncommand{x}",)


def test_numbers_units_acronyms_dates_ordinals_and_identifier_safety() -> None:
    profile = PronunciationProfile(language="es", acronym_policy="spell_out")
    text = (
        "0.05; 95%; 1.2 × 10^{-3}; 20–30; 1 kg; 2 kg; 2026-08-05; "
        "1.º; XYZ; 550e8400-e29b-41d4-a716-446655440000"
    )
    result = PronunciationEngine().transform(text, profile=profile)
    for expected in (
        "cero coma cero cinco",
        "noventa y cinco por ciento",
        "uno coma dos por diez elevado a menos tres",
        "veinte a treinta",
        "uno kilogramo",
        "dos kilogramos",
        "cinco de agosto de dos mil veinte y seis",
        "primero",
        "equis ye zeta",
        "550e8400-e29b-41d4-a716-446655440000",
    ):
        assert expected in result.spoken_text
    assert result.unsupported_fragments == ("XYZ",)
    assert any(item.rule_id == "builtin-unit-style" for item in result.applied_rules)


def test_preserve_policies_leave_numbers_units_and_unknown_acronyms_unchanged() -> None:
    profile = PronunciationProfile(
        language="en",
        acronym_policy="preserve",
        number_style="preserve",
        unit_style="preserve",
        punctuation_style="preserve",
    )
    result = PronunciationEngine().transform("0.05 2 kg XYZ", profile=profile)
    assert result.spoken_text == "0.05 2 kg XYZ"
    assert result.unsupported_fragments == ("XYZ",)
