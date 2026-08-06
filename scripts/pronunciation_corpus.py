#!/usr/bin/env python3
"""Validate and maintain the versioned pronunciation corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dialogue_studio.pronunciation.corpus import load_pronunciation_corpus
from dialogue_studio.pronunciation.corpus_maintenance import (
    add_candidate,
    corpus_statistics,
    deprecate_case,
    find_case,
    preview_case,
    promote_case,
    validate_corpus,
    write_corpus_report,
)

DEFAULT_CORPUS_ROOT = Path("tests/fixtures/pronunciation")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_CORPUS_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    stats_parser = commands.add_parser("stats")
    stats_parser.add_argument(
        "--report",
        type=Path,
        help="Escribe un reporte JSON nuevo en la ruta indicada",
    )

    list_parser = commands.add_parser("list")
    list_parser.add_argument("--status", choices=("approved", "candidate", "deprecated"))
    list_parser.add_argument("--language", choices=("es", "en"))

    show_parser = commands.add_parser("show")
    show_parser.add_argument("case_id")

    add_parser = commands.add_parser("add-candidate")
    add_parser.add_argument("--file", type=Path, required=True)

    for name in ("promote", "deprecate"):
        transition = commands.add_parser(name)
        transition.add_argument("case_id")
        transition.add_argument("--confirm", action="store_true")
    return parser


def _print_preview(root: Path, case_id: str) -> None:
    preview = preview_case(load_pronunciation_corpus(root), case_id)
    print(f"CASE: {preview.case.case_id}")
    print(f"STATUS: {preview.case.status}")
    print(f"PROFILE: {preview.case.profile}")
    print(f"WRITTEN:\n{preview.case.written_text}")
    print(f"CURRENT SPOKEN:\n{preview.current_spoken_text}")
    print(f"EXPECTED:\n{preview.case.expected_spoken_text}")
    print("WARNINGS: " + (", ".join(preview.warning_codes) or "none"))
    print("UNSUPPORTED: " + (", ".join(preview.unsupported_fragments) or "none"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root
    try:
        if args.command == "validate":
            snapshot = validate_corpus(root)
            print(f"Corpus válido: {len(snapshot.approved)} casos aprobados")
        elif args.command == "stats":
            snapshot = load_pronunciation_corpus(root)
            print(json.dumps(corpus_statistics(snapshot), ensure_ascii=False, indent=2))
            if args.report:
                destination = write_corpus_report(snapshot, args.report)
                print(f"Reporte escrito: {destination}", file=sys.stderr)
        elif args.command == "list":
            snapshot = load_pronunciation_corpus(root)
            for case in snapshot.cases:
                if args.status and case.status != args.status:
                    continue
                if args.language and case.language != args.language:
                    continue
                print(
                    f"{case.case_id}\t{case.status}\t{case.language}\t"
                    f"{case.profile}\t{case.category}"
                )
        elif args.command == "show":
            case = find_case(load_pronunciation_corpus(root), args.case_id)
            print(json.dumps(case.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        elif args.command == "add-candidate":
            case = add_candidate(root, args.file)
            print(f"Candidato agregado: {case.case_id}")
        elif args.command == "promote":
            _print_preview(root, args.case_id)
            case = promote_case(root, args.case_id, confirm=args.confirm)
            print(f"Caso promovido: {case.case_id}")
        elif args.command == "deprecate":
            _print_preview(root, args.case_id)
            case = deprecate_case(root, args.case_id, confirm=args.confirm)
            print(f"Caso deprecado: {case.case_id}")
    except (AssertionError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
