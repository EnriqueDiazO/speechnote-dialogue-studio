"""Position-preserving segmentation for prose, mathematics and protected text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SegmentKind = Literal[
    "prose",
    "math_inline",
    "math_display",
    "code",
    "url",
    "email",
    "path",
]
PROTECTED_SEGMENTS = {"code", "url", "email", "path"}


@dataclass(frozen=True)
class TextSegment:
    kind: SegmentKind
    text: str
    start: int
    end: int


_TOKEN_PATTERNS: tuple[tuple[SegmentKind, re.Pattern[str]], ...] = (
    ("code", re.compile(r"```[\s\S]*?```|`[^`\n]*`")),
    ("url", re.compile(r"https?://[^\s<>]+|www\.[^\s<>]+", re.IGNORECASE)),
    ("email", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    (
        "path",
        re.compile(
            r"(?<!\w)(?:[A-Za-z]:\\(?:[^\s\\]+\\)*[^\s\\]*|"
            r"/(?:[^\s/]+/)+[^\s/]*)"
        ),
    ),
    ("math_display", re.compile(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]")),
    ("math_inline", re.compile(r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)|\\\([^\n]*?\\\)")),
)

_UNICODE_OR_BARE_MATH = re.compile(
    r"(?<![\w\\])(?:"
    r"\\[A-Za-z]+(?:_\{?[^\s}]+\}?)?"
    r"|[A-Za-zΑ-ωϑϕ∇](?:_\{?[^\s}=]+\}?)?(?:\^\{?[^\s}=]+\}?)?"
    r")\s*(?:=|≠|≤|≥|≈|∼|∝|→|↦|∈|∉|⊂|⊆)\s*"
    r"[^\n,;!?]+"
)


def _split_bare_math(text: str, offset: int) -> list[TextSegment]:
    segments: list[TextSegment] = []
    cursor = 0
    for match in _UNICODE_OR_BARE_MATH.finditer(text):
        if match.start() > cursor:
            segments.append(
                TextSegment(
                    "prose",
                    text[cursor : match.start()],
                    offset + cursor,
                    offset + match.start(),
                )
            )
        raw = match.group(0)
        trimmed = raw.rstrip()
        if trimmed.endswith("."):
            trimmed = trimmed[:-1]
        end = match.start() + len(trimmed)
        segments.append(
            TextSegment("math_inline", trimmed, offset + match.start(), offset + end)
        )
        if end < match.end():
            segments.append(
                TextSegment("prose", text[end : match.end()], offset + end, offset + match.end())
            )
        cursor = match.end()
    if cursor < len(text):
        segments.append(TextSegment("prose", text[cursor:], offset + cursor, offset + len(text)))
    return segments


def segment_text(text: str) -> list[TextSegment]:
    """Split text without losing a character or changing original positions."""

    if not text:
        return []
    matches: list[tuple[int, int, int, SegmentKind, str]] = []
    for pattern_index, (kind, pattern) in enumerate(_TOKEN_PATTERNS):
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), pattern_index, kind, match.group(0)))
    matches.sort(key=lambda item: (item[0], item[2], -(item[1] - item[0])))
    selected: list[tuple[int, int, SegmentKind, str]] = []
    occupied_until = 0
    for start, end, _priority, kind, value in matches:
        if start < occupied_until:
            continue
        selected.append((start, end, kind, value))
        occupied_until = end

    segments: list[TextSegment] = []
    cursor = 0
    for start, end, kind, value in selected:
        if start > cursor:
            segments.extend(_split_bare_math(text[cursor:start], cursor))
        segments.append(TextSegment(kind, value, start, end))
        cursor = end
    if cursor < len(text):
        segments.extend(_split_bare_math(text[cursor:], cursor))
    return [segment for segment in segments if segment.text]
