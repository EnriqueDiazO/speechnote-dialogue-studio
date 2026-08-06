"""Deterministic lightweight handling of numbers, units, acronyms and punctuation."""

from __future__ import annotations

import re

from .models import AppliedPronunciationRule, PronunciationProfile, PronunciationWarning

_ONES_ES = (
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
    "dieciocho", "diecinueve",
)
_TENS_ES = (
    "", "", "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta",
    "ochenta", "noventa",
)
_ONES_EN = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
)
_TENS_EN = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_LETTERS = {
    "es": {**{key: value for key, value in zip("abcdefghijklmnñopqrstuvwxyz", (
        "a", "be", "ce", "de", "e", "efe", "ge", "hache", "i", "jota", "ka", "ele",
        "eme", "ene", "eñe", "o", "pe", "cu", "erre", "ese", "te", "u", "uve",
        "doble u", "equis", "ye", "zeta",
    ), strict=True)}},
    "en": {key: value for key, value in zip("abcdefghijklmnopqrstuvwxyz", (
        "ay", "bee", "see", "dee", "ee", "eff", "gee", "aitch", "eye", "jay", "kay",
        "el", "em", "en", "oh", "pee", "cue", "ar", "ess", "tee", "you", "vee",
        "double you", "ex", "why", "zee",
    ), strict=True)},
}


def _integer_words(value: int, language: str) -> str:
    ones = _ONES_EN if language == "en" else _ONES_ES
    tens = _TENS_EN if language == "en" else _TENS_ES
    if value < 0:
        return ("minus " if language == "en" else "menos ") + _integer_words(-value, language)
    if value < 20:
        return ones[value]
    if value < 100:
        quotient, remainder = divmod(value, 10)
        if not remainder:
            return tens[quotient]
        connector = "-" if language == "en" else " y "
        return tens[quotient] + connector + ones[remainder]
    if value < 1_000:
        quotient, remainder = divmod(value, 100)
        if language == "en":
            base = f"{ones[quotient]} hundred"
        elif quotient == 1:
            base = "cien" if not remainder else "ciento"
        elif quotient == 5:
            base = "quinientos"
        elif quotient == 7:
            base = "setecientos"
        elif quotient == 9:
            base = "novecientos"
        else:
            base = ones[quotient] + "cientos"
        return base if not remainder else f"{base} {_integer_words(remainder, language)}"
    if value < 10_000:
        quotient, remainder = divmod(value, 1_000)
        base = "one thousand" if language == "en" and quotient == 1 else (
            "mil" if language == "es" and quotient == 1 else (
                f"{_integer_words(quotient, language)} thousand"
                if language == "en"
                else f"{_integer_words(quotient, language)} mil"
            )
        )
        return base if not remainder else f"{base} {_integer_words(remainder, language)}"
    return str(value)


def _number_words(token: str, language: str, style: str) -> str:
    if style == "preserve":
        return token
    if style == "digits":
        separator = " point " if language == "en" else " coma "
        return separator.join(
            " ".join(_integer_words(int(char), language) for char in part)
            for part in token.split(".")
        )
    if "." in token:
        integer, decimal = token.split(".", 1)
        separator = "point" if language == "en" else "coma"
        digits = " ".join(_integer_words(int(char), language) for char in decimal)
        return f"{_integer_words(int(integer), language)} {separator} {digits}"
    return _integer_words(int(token), language)


def apply_linguistic_styles(
    text: str,
    profile: PronunciationProfile,
    *,
    known_acronyms: set[str],
    units: dict[str, str],
    source_offset: int = 0,
) -> tuple[str, list[AppliedPronunciationRule], list[PronunciationWarning], list[str]]:
    language = "en" if profile.language.startswith("en") else "es"
    applied: list[AppliedPronunciationRule] = []
    warnings: list[PronunciationWarning] = []
    unsupported: list[str] = []

    def traced_replace(
        rule_id: str,
        kind: str,
        match: re.Match[str],
        replacement: str,
    ) -> str:
        applied.append(
            AppliedPronunciationRule(
                rule_id=rule_id,
                kind=kind,
                scope="builtin",
                source_span=(
                    source_offset + match.start(),
                    source_offset + match.end(),
                ),
                source_text=match.group(0),
                replacement_text=replacement,
                priority=-100,
            )
        )
        return replacement

    scientific_pattern = re.compile(
        r"(?<![\w-])(\d+(?:\.\d+)?)\s*[×x]\s*10\^\{?(-?\d+)\}?(?![\w-])"
    )

    def scientific_replacement(match: re.Match[str]) -> str:
        coefficient = _number_words(match.group(1), language, profile.number_style)
        exponent = _number_words(match.group(2), language, profile.number_style)
        replacement = (
            f"{coefficient} times ten to the power {exponent}"
            if language == "en"
            else f"{coefficient} por diez elevado a {exponent}"
        )
        return traced_replace("builtin-scientific-notation", "number", match, replacement)

    if profile.number_style != "preserve":
        text = scientific_pattern.sub(scientific_replacement, text)

    range_pattern = re.compile(r"(?<![\w-])(\d{1,4})[–—-](\d{1,4})(?![\w-])")

    def range_replacement(match: re.Match[str]) -> str:
        start = _number_words(match.group(1), language, profile.number_style)
        end = _number_words(match.group(2), language, profile.number_style)
        connector = "to" if language == "en" else "a"
        return traced_replace(
            "builtin-number-range", "number", match, f"{start} {connector} {end}"
        )

    if profile.number_style != "preserve":
        text = range_pattern.sub(range_replacement, text)

    date_pattern = re.compile(
        r"(?<![\w-])(?:(\d{4})-(\d{1,2})-(\d{1,2})|(\d{1,2})/(\d{1,2})/(\d{4}))(?![\w-])"
    )

    def date_replacement(match: re.Match[str]) -> str:
        if match.group(1):
            year, month, day = match.group(1), match.group(2), match.group(3)
        else:
            day, month, year = match.group(4), match.group(5), match.group(6)
        assert year is not None and month is not None and day is not None
        spoken_day = _number_words(day, language, profile.number_style)
        spoken_year = _number_words(year, language, profile.number_style)
        months = (
            (
                "January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December",
            )
            if language == "en"
            else (
                "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
                "agosto", "septiembre", "octubre", "noviembre", "diciembre",
            )
        )
        month_number = int(month)
        if not 1 <= month_number <= 12:
            return match.group(0)
        spoken_month = months[month_number - 1]
        replacement = (
            f"{spoken_month} {spoken_day}, {spoken_year}"
            if language == "en"
            else f"{spoken_day} de {spoken_month} de {spoken_year}"
        )
        return traced_replace("builtin-date-style", "number", match, replacement)

    if profile.number_style != "preserve":
        text = date_pattern.sub(date_replacement, text)

    ordinal_pattern = re.compile(
        r"(?<![\w-])(\d{1,3})(?:st|nd|rd|th|\.?(?:º|ª))(?![\w-])",
        re.IGNORECASE,
    )

    def ordinal_replacement(match: re.Match[str]) -> str:
        value = int(match.group(1))
        basic_es = {1: "primero", 2: "segundo", 3: "tercero", 4: "cuarto", 5: "quinto"}
        basic_en = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
        basic = basic_en if language == "en" else basic_es
        replacement = basic.get(
            value,
            f"{_number_words(str(value), language, profile.number_style)} ordinal",
        )
        return traced_replace("builtin-ordinal-style", "number", match, replacement)

    if profile.number_style != "preserve":
        text = ordinal_pattern.sub(ordinal_replacement, text)

    percent_pattern = re.compile(r"(?<![\w-])(\d+(?:\.\d+)?)\s*%(?!\w)")

    def percent_replacement(match: re.Match[str]) -> str:
        value = _number_words(match.group(1), language, profile.number_style)
        suffix = "percent" if language == "en" else "por ciento"
        return traced_replace(
            "builtin-percentage", "number", match, f"{value} {suffix}"
        )

    if profile.number_style != "preserve":
        text = percent_pattern.sub(percent_replacement, text)

    if profile.unit_style != "preserve" and units:
        unit_pattern = re.compile(
            r"(?<![\w-])(\d+(?:\.\d+)?)\s*("
            + "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))
            + r")(?!\w)"
        )

        def unit_replacement(match: re.Match[str]) -> str:
            raw_value = match.group(1)
            value = _number_words(raw_value, language, profile.number_style)
            label = units[match.group(2)]
            if raw_value in {"1", "1.0"}:
                if label.lower().startswith("grados "):
                    label = "grado " + label[7:]
                elif label.endswith("s") and not label.lower().endswith("hertz"):
                    label = label[:-1]
            return traced_replace(
                "builtin-unit-style", "unit", match, f"{value} {label}"
            )

        text = unit_pattern.sub(unit_replacement, text)

    if profile.number_style != "preserve":
        number_pattern = re.compile(r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])")

        def number_replacement(match: re.Match[str]) -> str:
            replacement = _number_words(match.group(0), language, profile.number_style)
            return traced_replace(
                "builtin-number-style", "number", match, replacement
            )

        text = number_pattern.sub(number_replacement, text)

    unknown_pattern = re.compile(
        r"(?<![\w-])(?:[A-ZÁÉÍÓÚÑ]{2,}\d*|[A-Z][a-z]+[A-Z]\w*)(?![\w-])"
    )

    def acronym_replacement(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in known_acronyms:
            return token
        unsupported.append(token)
        warnings.append(
            PronunciationWarning(
                code="unknown_acronym",
                message=f"Sigla o término técnico por revisar: {token}",
                source_span=match.span(),
                fragment=token,
            )
        )
        if profile.acronym_policy in {"custom", "preserve"}:
            return token
        if profile.acronym_policy == "read_as_word":
            return token.lower()
        letters = _LETTERS[language]
        replacement = " ".join(letters.get(char.lower(), char) for char in token)
        return traced_replace(
            "builtin-acronym-policy", "acronym", match, replacement
        )

    text = unknown_pattern.sub(acronym_replacement, text)
    return text, applied, warnings, unsupported
