from __future__ import annotations

import threading
from pathlib import Path

import pytest

from dialogue_studio.audio import AudioError
from dialogue_studio.speechnote import SpeechNoteError
from dialogue_studio.synthesis import (
    SynthesisBusyError,
    SynthesisCoordinator,
    run_with_synthesis_state,
)


def test_success_clears_transient_synthesis_state(tmp_path: Path) -> None:
    coordinator = SynthesisCoordinator()
    result = run_with_synthesis_state(
        coordinator,
        "utterance",
        tmp_path / "output.wav",
        "session",
        lambda: "ok",
    )
    assert result == "ok"
    assert coordinator.active is None


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timeout"),
        SpeechNoteError("error CLI"),
        AudioError("error ffprobe"),
        SpeechNoteError("WAV inválido"),
        RuntimeError("excepción inesperada"),
        KeyboardInterrupt(),
    ],
    ids=["timeout", "cli", "ffprobe", "invalid-wav", "unexpected", "cancellation"],
)
def test_every_failure_path_clears_transient_state(error: BaseException, tmp_path: Path) -> None:
    coordinator = SynthesisCoordinator()

    def fail() -> None:
        raise error

    with pytest.raises(type(error)):
        run_with_synthesis_state(
            coordinator,
            "utterance",
            tmp_path / "output.wav",
            "session",
            fail,
        )
    assert coordinator.active is None
    assert (
        run_with_synthesis_state(
            coordinator,
            "next",
            tmp_path / "next.wav",
            "session",
            lambda: "next-ok",
        )
        == "next-ok"
    )


def test_two_syntheses_cannot_run_concurrently(tmp_path: Path) -> None:
    coordinator = SynthesisCoordinator()
    with coordinator.track("first", tmp_path / "first.wav", "session-a"):
        assert coordinator.active is not None
        with pytest.raises(SynthesisBusyError, match="activa"):
            coordinator.start("second", tmp_path / "second.wav", "session-b")
    assert coordinator.active is None


def test_abandoned_owner_thread_does_not_revive_a_lock(tmp_path: Path) -> None:
    coordinator = SynthesisCoordinator()

    def abandon_marker() -> None:
        coordinator.start("first", tmp_path / "first.wav", "session")

    thread = threading.Thread(target=abandon_marker)
    thread.start()
    thread.join()
    assert coordinator.active is None
    coordinator.start("second", tmp_path / "second.wav", "session")
    assert coordinator.active is not None
    coordinator.clear(coordinator.active.job_token)
