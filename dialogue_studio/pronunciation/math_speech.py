"""Limited recursive LaTeX and Unicode mathematics verbalizer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .linguistics import _number_words
from .models import MathStyle, PronunciationWarning


@dataclass(frozen=True)
class MathSpeechResult:
    spoken_text: str
    warnings: tuple[PronunciationWarning, ...]
    unsupported_fragments: tuple[str, ...]


_FUNCTIONS = {
    "es": {
        "sin": "seno",
        "cos": "coseno",
        "tan": "tangente",
        "log": "logaritmo",
        "ln": "logaritmo natural",
        "exp": "exponencial",
        "max": "máximo",
        "min": "mínimo",
        "det": "determinante",
        "tr": "traza",
        "rank": "rango",
        "ker": "núcleo",
        "im": "imagen",
    },
    "en": {
        "sin": "sine",
        "cos": "cosine",
        "tan": "tangent",
        "log": "logarithm",
        "ln": "natural logarithm",
        "exp": "exponential",
        "max": "maximum",
        "min": "minimum",
        "det": "determinant",
        "tr": "trace",
        "rank": "rank",
        "ker": "kernel",
        "im": "image",
    },
}

_LETTERS_ES = {
    "a": "a",
    "b": "be",
    "c": "ce",
    "d": "de",
    "e": "e",
    "f": "efe",
    "g": "ge",
    "h": "hache",
    "i": "i",
    "j": "jota",
    "k": "ka",
    "l": "ele",
    "m": "eme",
    "n": "ene",
    "o": "o",
    "p": "pe",
    "q": "cu",
    "r": "erre",
    "s": "ese",
    "t": "te",
    "u": "u",
    "v": "uve",
    "w": "doble u",
    "x": "equis",
    "y": "ye",
    "z": "zeta",
}

_OPERATOR_FALLBACK = {
    "es": {
        "+": "más",
        "-": "menos",
        "−": "menos",
        "*": "por",
        "×": "por",
        "·": "por",
        "/": "dividido entre",
        "=": "igual a",
        "<": "menor que",
        ">": "mayor que",
        "≤": "menor o igual que",
        "≥": "mayor o igual que",
        "≈": "aproximadamente igual a",
        "∼": "similar a",
        "±": "más o menos",
        "∓": "menos o más",
        "∝": "proporcional a",
        "→": "tiende a",
        "↦": "se transforma en",
        "∈": "pertenece a",
        "∉": "no pertenece a",
        "⊂": "es subconjunto de",
        "⊆": "es subconjunto o igual a",
        "∪": "unión",
        "∩": "intersección",
    },
    "en": {
        "+": "plus",
        "-": "minus",
        "−": "minus",
        "*": "times",
        "×": "times",
        "·": "times",
        "/": "divided by",
        "=": "equals",
        "<": "less than",
        ">": "greater than",
        "≤": "less than or equal to",
        "≥": "greater than or equal to",
        "≈": "approximately equal to",
        "∼": "similar to",
        "±": "plus or minus",
        "∓": "minus or plus",
        "∝": "proportional to",
        "→": "approaches",
        "↦": "maps to",
        "∈": "belongs to",
        "∉": "does not belong to",
        "⊂": "is a subset of",
        "⊆": "is a subset of or equal to",
        "∪": "union",
        "∩": "intersection",
    },
}


def strip_math_delimiters(value: str) -> str:
    pairs = (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    stripped = value.strip()
    for opening, closing in pairs:
        if stripped.startswith(opening) and stripped.endswith(closing):
            return stripped[len(opening) : -len(closing)].strip()
    return stripped


class _Reader:
    def __init__(
        self,
        text: str,
        *,
        language: str,
        style: MathStyle,
        aliases: dict[str, str],
        number_style: str,
        source_offset: int = 0,
    ) -> None:
        self.text = text
        self.language = "en" if language.startswith("en") else "es"
        self.style = style
        self.aliases = aliases
        self.number_style = number_style
        self.source_offset = source_offset
        self.position = 0
        self.warnings: list[PronunciationWarning] = []
        self.unsupported: list[str] = []

    def _word(self, es: str, en: str) -> str:
        return en if self.language == "en" else es

    def _subreader(self, value: str) -> str:
        reader = _Reader(
            value,
            language=self.language,
            style=self.style,
            aliases=self.aliases,
            number_style=self.number_style,
            source_offset=self.source_offset,
        )
        spoken = reader.read()
        self.warnings.extend(reader.warnings)
        self.unsupported.extend(reader.unsupported)
        return spoken

    def _balanced(self, opening: str, closing: str) -> str | None:
        if not self.text.startswith(opening, self.position):
            return None
        start = self.position + len(opening)
        cursor = start
        depth = 1
        while cursor < len(self.text):
            if self.text.startswith(opening, cursor):
                depth += 1
                cursor += len(opening)
                continue
            if self.text.startswith(closing, cursor):
                depth -= 1
                if depth == 0:
                    self.position = cursor + len(closing)
                    return self.text[start:cursor]
                cursor += len(closing)
                continue
            cursor += 1
        return None

    def _argument(self) -> str:
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1
        grouped = self._balanced("{", "}")
        if grouped is not None:
            return grouped
        if self.position >= len(self.text):
            return ""
        if self.text[self.position] == "\\":
            start = self.position
            self.position += 1
            while self.position < len(self.text) and self.text[self.position].isalpha():
                self.position += 1
            return self.text[start : self.position]
        start = self.position
        if self.text[self.position] in "+-":
            self.position += 1
        while self.position < len(self.text) and self.text[self.position].isalnum():
            self.position += 1
        if self.position == start:
            self.position += 1
        return self.text[start : self.position]

    def _scripts(self) -> tuple[str | None, str | None]:
        subscript: str | None = None
        superscript: str | None = None
        while self.position < len(self.text) and self.text[self.position] in "_^":
            marker = self.text[self.position]
            self.position += 1
            value = self._argument()
            if marker == "_":
                subscript = value
            else:
                superscript = value
        return subscript, superscript

    def _with_scripts(self, base: str) -> str:
        subscript, superscript = self._scripts()
        parts = [base]
        if subscript is not None:
            spoken = self._subreader(subscript)
            parts.append(self._word(f"sub {spoken}", f"sub {spoken}"))
        if superscript is not None:
            spoken = self._subreader(superscript)
            if superscript.strip() == "2":
                parts.append(self._word("al cuadrado", "squared"))
            elif superscript.strip() == "3":
                parts.append(self._word("al cubo", "cubed"))
            elif superscript.replace(" ", "") == "-1":
                parts.append(self._word("inversa", "inverse"))
            elif superscript.strip().upper() == "T":
                parts.append(self._word("transpuesta", "transpose"))
            else:
                parts.append(self._word(f"elevado a {spoken}", f"to the power {spoken}"))
        return " ".join(parts)

    def _identifier(self) -> str:
        start = self.position
        while self.position < len(self.text) and (
            self.text[self.position].isalnum() or self.text[self.position] == "."
        ):
            self.position += 1
        value = self.text[start : self.position]
        if value in self.aliases:
            spoken = self.aliases[value]
        elif value.lower() == "softmax":
            spoken = "soft max"
        elif value in _FUNCTIONS[self.language]:
            spoken = _FUNCTIONS[self.language][value]
        elif value.isdigit() or re.fullmatch(r"\d+(?:\.\d+)?", value):
            spoken = _number_words(value, self.language, self.number_style)
        elif len(value) == 1 and self.language == "es":
            spoken = _LETTERS_ES.get(value.lower(), value)
        elif len(value) == 2 and value[0].lower() == "d" and value[1].isalpha():
            letter = _LETTERS_ES.get(value[1].lower(), value[1])
            spoken = self._word(f"diferencial de {letter}", f"d {value[1]}")
        elif value.isalpha() and len(value) <= 3:
            if self.language == "es":
                spoken = " ".join(_LETTERS_ES.get(letter.lower(), letter) for letter in value)
            else:
                spoken = " ".join(value)
        else:
            spoken = value
        spoken = self._with_scripts(spoken)
        if self.position < len(self.text) and self.text[self.position] == "(":
            grouped = self._balanced("(", ")")
            if grouped is not None:
                argument = self._subreader(grouped)
                return self._word(f"{spoken} de {argument}", f"{spoken} of {argument}")
        return spoken

    def _fraction(self) -> str:
        numerator = self._subreader(self._argument())
        denominator = self._subreader(self._argument())
        if self.style == "concise":
            return self._word(
                f"{numerator} sobre {denominator}",
                f"{numerator} over {denominator}",
            )
        if self.style == "explicit":
            return self._word(
                f"fracción con numerador {numerator} y denominador {denominator}",
                f"fraction with numerator {numerator} and denominator {denominator}",
            )
        if self.style == "symbolic":
            return self._word(
                f"fracción {numerator}, {denominator}",
                f"fraction {numerator}, {denominator}",
            )
        return self._word(
            f"{numerator} dividido entre {denominator}",
            f"{numerator} divided by {denominator}",
        )

    def _root(self) -> str:
        index = None
        if self.position < len(self.text) and self.text[self.position] == "[":
            index = self._balanced("[", "]")
        radicand = self._subreader(self._argument())
        if index is None:
            return self._word(f"raíz cuadrada de {radicand}", f"square root of {radicand}")
        spoken_index = self._subreader(index)
        return self._word(
            f"raíz de índice {spoken_index} de {radicand}",
            f"root with index {spoken_index} of {radicand}",
        )

    def _bounded_operator(self, command: str) -> str:
        subscript, superscript = self._scripts()
        names = {
            "sum": ("sumatoria", "sum"),
            "prod": ("producto", "product"),
            "int": ("integral", "integral"),
            "iint": ("integral doble", "double integral"),
            "iiint": ("integral triple", "triple integral"),
            "oint": ("integral de contorno", "contour integral"),
        }
        base = self._word(*names[command])
        parts = [base]
        if subscript:
            lower = self._subreader(subscript)
            parts.append(self._word(f"desde {lower}", f"from {lower}"))
        if superscript:
            upper = self._subreader(superscript)
            parts.append(self._word(f"hasta {upper}", f"to {upper}"))
        parts.append(self._word("de", "of"))
        return " ".join(parts)

    def _limit(self) -> str:
        subscript, _superscript = self._scripts()
        if not subscript:
            return self._word("límite", "limit")
        approach = self._subreader(subscript)
        return self._word(f"límite cuando {approach}", f"limit as {approach}")

    def _unknown(self, start: int, command: str) -> str:
        argument = ""
        saved = self.position
        raw_argument = self._argument()
        if raw_argument:
            argument = self._subreader(raw_argument)
        else:
            self.position = saved
        fragment = self.text[start : self.position]
        self.unsupported.append(fragment)
        self.warnings.append(
            PronunciationWarning(
                code="unsupported_math_command",
                message=f"Comando matemático no soportado: \\{command}",
                source_span=(self.source_offset + start, self.source_offset + self.position),
                fragment=fragment,
            )
        )
        readable = command.replace("_", " ")
        return f"{readable} {argument}".strip()

    def _command(self) -> str:
        start = self.position
        self.position += 1
        if self.position < len(self.text) and not self.text[self.position].isalpha():
            spacing = self.text[self.position]
            self.position += 1
            return "" if spacing in ",;! " else spacing
        name_start = self.position
        while self.position < len(self.text) and self.text[self.position].isalpha():
            self.position += 1
        command = self.text[name_start : self.position]
        raw = f"\\{command}"
        if raw in self.aliases:
            return self._with_scripts(self.aliases[raw])
        if command == "frac":
            return self._fraction()
        if command == "sqrt":
            return self._root()
        if command in {"sum", "prod", "int", "iint", "iiint", "oint"}:
            return self._bounded_operator(command)
        if command == "lim":
            return self._limit()
        if command in _FUNCTIONS[self.language]:
            return self._with_scripts(_FUNCTIONS[self.language][command])
        if command == "arg" and self.text.startswith("\\max", self.position):
            self.position += len("\\max")
            return self._word("argumento que maximiza", "argument that maximizes")
        if command == "arg" and self.text.startswith("\\min", self.position):
            self.position += len("\\min")
            return self._word("argumento que minimiza", "argument that minimizes")
        if command == "operatorname":
            return self._subreader(self._argument())
        if command in {"vec", "mathbf"}:
            content = self._subreader(self._argument())
            if command == "vec":
                return self._word(f"vector {content}", f"vector {content}")
            return self._word(f"{content} en negrita", f"bold {content}")
        if command in {"mathbb", "mathrm", "mathcal"}:
            raw_content = self._argument()
            if command == "mathbb":
                spaces = {
                    "R": ("números reales", "real numbers"),
                    "C": ("números complejos", "complex numbers"),
                    "N": ("números naturales", "natural numbers"),
                    "Z": ("números enteros", "integers"),
                }
                if raw_content in spaces:
                    return self._word(*spaces[raw_content])
            return self._subreader(raw_content)
        if command == "hat":
            content = self._subreader(self._argument())
            return self._word(f"{content} estimada", f"{content} hat")
        if command == "langle":
            closing = self.text.find("\\rangle", self.position)
            if closing >= 0:
                content = self._subreader(self.text[self.position:closing])
                self.position = closing + len("\\rangle")
                return self._word(f"producto interno de {content}", f"inner product of {content}")
        if command in {"left", "right", "limits"}:
            return ""
        if command == "begin":
            environment = self._argument()
            closing = rf"\end{{{environment}}}"
            end = self.text.find(closing, self.position)
            if end >= 0 and environment in {"matrix", "pmatrix", "bmatrix", "vmatrix"}:
                content = self.text[self.position:end]
                self.position = end + len(closing)
                rows = [row for row in content.split(r"\\") if row.strip()]
                spoken_rows = [
                    self._word(
                        "fila "
                        + " columna ".join(
                            self._subreader(cell) for cell in row.split("&")
                        ),
                        "row "
                        + " column ".join(
                            self._subreader(cell) for cell in row.split("&")
                        ),
                    )
                    for row in rows
                ]
                return self._word(
                    "matriz " + "; ".join(spoken_rows),
                    "matrix " + "; ".join(spoken_rows),
                )
        return self._unknown(start, command)

    def _atom(self) -> str:
        character = self.text[self.position]
        if character == "\\":
            if self.text.startswith(r"\|", self.position):
                self.position += 2
                closing = self.text.find(r"\|", self.position)
                if closing >= 0:
                    content = self._subreader(self.text[self.position:closing])
                    self.position = closing + 2
                    return self._word(f"norma de {content}", f"norm of {content}")
            spoken = self._command()
            if self.position < len(self.text) and self.text[self.position] == "(":
                grouped = self._balanced("(", ")")
                if grouped is not None:
                    argument = self._subreader(grouped)
                    if spoken.endswith((" de", " of")):
                        spoken = f"{spoken} {argument}"
                    else:
                        spoken = self._word(
                            f"{spoken} de {argument}",
                            f"{spoken} of {argument}",
                        )
            return self._with_scripts(spoken)
        if character in self.aliases:
            self.position += 1
            return self._with_scripts(self.aliases[character])
        if character.isalnum() or character.isalpha():
            return self._identifier()
        if self.text.startswith("||", self.position):
            self.position += 2
            closing = self.text.find("||", self.position)
            if closing >= 0:
                content = self._subreader(self.text[self.position:closing])
                self.position = closing + 2
                return self._word(f"norma de {content}", f"norm of {content}")
        if character == "|":
            self.position += 1
            closing = self.text.find("|", self.position)
            if closing >= 0:
                content = self._subreader(self.text[self.position:closing])
                self.position = closing + 1
                return self._word(f"valor absoluto de {content}", f"absolute value of {content}")
        if character in "({[":
            closing = {"(": ")", "{": "}", "[": "]"}[character]
            content = self._balanced(character, closing)
            if content is not None:
                spoken = self._subreader(content)
                if self.style in {"explicit", "symbolic"}:
                    names = {
                        "(": ("paréntesis", "parentheses"),
                        "[": ("corchetes", "brackets"),
                        "{": ("llaves", "braces"),
                    }
                    label = self._word(*names[character])
                    grouped_spoken = self._word(
                        f"abre {label} {spoken} cierra {label}",
                        f"open {label} {spoken} close {label}",
                    )
                    return self._with_scripts(grouped_spoken)
                return self._with_scripts(spoken)
        operator = self.aliases.get(character) or _OPERATOR_FALLBACK[self.language].get(character)
        self.position += 1
        return operator if operator is not None else character

    def read(self) -> str:
        tokens: list[str] = []
        while self.position < len(self.text):
            if self.text[self.position].isspace():
                self.position += 1
                continue
            token = self._atom().strip()
            if token:
                tokens.append(token)
        spoken = " ".join(tokens)
        spoken = re.sub(r"\s+", " ", spoken).strip()
        spoken = re.sub(r"\s+([,.;:])", r"\1", spoken)
        return spoken


def verbalize_math(
    expression: str,
    *,
    language: str,
    style: MathStyle,
    aliases: dict[str, str] | None = None,
    number_style: str = "natural",
    source_offset: int = 0,
) -> MathSpeechResult:
    source = strip_math_delimiters(expression)
    reader = _Reader(
        source,
        language=language,
        style=style,
        aliases=aliases or {},
        number_style=number_style,
        source_offset=source_offset,
    )
    spoken = reader.read()
    return MathSpeechResult(
        spoken_text=spoken or source,
        warnings=tuple(reader.warnings),
        unsupported_fragments=tuple(reader.unsupported),
    )
