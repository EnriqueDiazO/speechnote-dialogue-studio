"""Process-local coordination for Speech Note synthesis jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .models import utc_now


class SynthesisBusyError(RuntimeError):
    """A second synthesis was requested while a real job is active."""


@dataclass(frozen=True)
class ActiveSynthesis:
    utterance_id: str
    started_at: str
    output_path: Path
    session_token: str
    job_token: str
    owner_thread_id: int


class SynthesisCoordinator:
    """Track one real synthesis in memory without persisting a lock."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._active: ActiveSynthesis | None = None

    def _owner_is_alive(self, active: ActiveSynthesis) -> bool:
        return any(
            thread.ident == active.owner_thread_id and thread.is_alive()
            for thread in threading.enumerate()
        )

    def clear_abandoned(self) -> bool:
        """Clear only a marker whose owner thread no longer exists."""
        with self._guard:
            if self._active is None or self._owner_is_alive(self._active):
                return False
            self._active = None
            return True

    @property
    def active(self) -> ActiveSynthesis | None:
        self.clear_abandoned()
        with self._guard:
            return self._active

    def start(
        self,
        utterance_id: str,
        output_path: Path,
        session_token: str,
    ) -> ActiveSynthesis:
        self.clear_abandoned()
        with self._guard:
            if self._active is not None:
                raise SynthesisBusyError("Hay una síntesis real activa")
            active = ActiveSynthesis(
                utterance_id=utterance_id,
                started_at=utc_now(),
                output_path=output_path,
                session_token=session_token,
                job_token=uuid4().hex,
                owner_thread_id=threading.get_ident(),
            )
            self._active = active
            return active

    def clear(self, job_token: str) -> None:
        with self._guard:
            if self._active is not None and self._active.job_token == job_token:
                self._active = None

    @contextmanager
    def track(
        self,
        utterance_id: str,
        output_path: Path,
        session_token: str,
    ) -> Iterator[ActiveSynthesis]:
        active = self.start(utterance_id, output_path, session_token)
        try:
            yield active
        finally:
            self.clear(active.job_token)


def run_with_synthesis_state(
    coordinator: SynthesisCoordinator,
    utterance_id: str,
    output_path: Path,
    session_token: str,
    operation: Callable[[], object],
) -> object:
    """Run an operation while guaranteeing transient-state cleanup."""
    with coordinator.track(utterance_id, output_path, session_token):
        return operation()


GLOBAL_SYNTHESIS_COORDINATOR = SynthesisCoordinator()
