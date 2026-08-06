from __future__ import annotations

from pathlib import Path

from dialogue_studio.pronunciation.corpus import load_pronunciation_corpus

CORPUS_ROOT = Path("tests/fixtures/pronunciation")


def test_corpus_discovers_versioned_files_without_fixed_file_list() -> None:
    snapshot = load_pronunciation_corpus(CORPUS_ROOT)
    assert snapshot.manifest.corpus_version == "1.0.0"
    assert len(snapshot.candidates) == 5
    assert len(snapshot.approved) == 108
    assert not snapshot.deprecated
    assert {case.written_text for case in snapshot.candidates} == {
        "Haseman",
        "Fredholm",
        "Wiener–Hopf",
        "Mellin",
        "Calkin",
    }
    assert all(not case.expected_spoken_text for case in snapshot.candidates)
    assert all(snapshot.case_paths[case.case_id].is_file() for case in snapshot.cases)
