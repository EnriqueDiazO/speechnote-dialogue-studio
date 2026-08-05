from __future__ import annotations

from pathlib import Path

from dialogue_studio.paths import AppPaths
from dialogue_studio.qwen_preview import (
    clear_qwen_previews,
    generate_qwen_previews,
    preview_fingerprint,
)
from dialogue_studio.synthesis import SynthesisCoordinator


class FakeClient:
    def __init__(self, make_wav) -> None:
        self.make_wav = make_wav
        self.calls: list[str] = []

    def synthesize(self, **kwargs):
        voice = kwargs["speaker"]
        self.calls.append(voice)
        self.make_wav(kwargs["output_path"], rate=24_000, duration=0.2)
        return {"elapsed_seconds": len(self.calls) / 10}


def test_preview_fingerprint_covers_voice_text_language_and_sampling() -> None:
    base = preview_fingerprint("Hola", "serena", "spanish", {"seed": 1})
    assert base != preview_fingerprint("Adiós", "serena", "spanish", {"seed": 1})
    assert base != preview_fingerprint("Hola", "vivian", "spanish", {"seed": 1})
    assert base != preview_fingerprint("Hola", "serena", "english", {"seed": 1})
    assert base != preview_fingerprint("Hola", "serena", "spanish", {"seed": 2})


def test_previews_generate_sequentially_cache_and_do_not_touch_project_audio(
    make_wav, tmp_path: Path
) -> None:
    paths = AppPaths(tmp_path / "Music")
    paths.ensure()
    client = FakeClient(make_wav)
    coordinator = SynthesisCoordinator()
    options = {"seed": 1, "temperature": 0.9}
    first = generate_qwen_previews(
        paths=paths,
        text="El mismo texto",
        voice_ids=["serena", "vivian", "ryan"],
        language="spanish",
        generation_options=options,
        session_token="session",
        coordinator=coordinator,
        client=client,
    )
    assert client.calls == ["serena", "vivian", "ryan"]
    assert all(item.path and item.path.is_file() for item in first)
    assert all(item.duration_seconds == 0.2 for item in first)
    assert coordinator.active is None
    assert not list(paths.projects.rglob("*.wav"))

    second = generate_qwen_previews(
        paths=paths,
        text="El mismo texto",
        voice_ids=["serena", "vivian", "ryan"],
        language="spanish",
        generation_options=options,
        session_token="session",
        coordinator=coordinator,
        client=client,
    )
    assert client.calls == ["serena", "vivian", "ryan"]
    assert all(item.cached for item in second)
    assert clear_qwen_previews(paths) == 3
    assert not list(paths.temporary.rglob("*.wav"))


def test_invalid_cached_preview_is_preserved_as_partial(make_wav, tmp_path: Path) -> None:
    paths = AppPaths(tmp_path / "Music")
    paths.ensure()
    client = FakeClient(make_wav)
    fingerprint = preview_fingerprint("Hola", "serena", "spanish", {"seed": 1})
    invalid = (
        paths.temporary / "qwen-previews" / f"{fingerprint[:24]}-serena.wav"
    )
    invalid.parent.mkdir(parents=True)
    invalid.write_bytes(b"corrupt")
    result = generate_qwen_previews(
        paths=paths,
        text="Hola",
        voice_ids=["serena"],
        language="spanish",
        generation_options={"seed": 1},
        session_token="session",
        coordinator=SynthesisCoordinator(),
        client=client,
    )
    assert result[0].path and result[0].path.is_file()
    assert list(invalid.parent.glob("*.partial"))
