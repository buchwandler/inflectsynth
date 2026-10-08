from __future__ import annotations

import json
from typing import Any

import pytest

import inflectsynth.__main__ as cli
from inflectsynth import DEFAULT_MODEL, DEFAULT_VOICE, RequestMeasure


class FakeVoice:
    def __init__(self) -> None:
        self.measured_text: str | None = None

    def __enter__(self) -> FakeVoice:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def measure_prepared(self, text: str) -> RequestMeasure:
        self.measured_text = text
        return RequestMeasure(
            fits=None,
            amount=42,
            maximum=None,
            model_id=DEFAULT_MODEL,
        )


def test_cli_defaults_use_public_constants() -> None:
    args = cli.parse_args([])
    assert args.model == DEFAULT_MODEL
    assert args.voice == DEFAULT_VOICE


def test_cli_measure_outputs_json_without_synthesis(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    voice = FakeVoice()
    opened: dict[str, Any] = {}

    def open_voice(model: str, **kwargs: Any) -> FakeVoice:
        opened.update(model=model, **kwargs)
        return voice

    monkeypatch.setattr(cli.InflectVoice, "from_pretrained", open_voice)

    assert cli.main(["--measure", "  exact prepared text  "]) == 0

    assert voice.measured_text == "  exact prepared text  "
    assert opened["model"] == DEFAULT_MODEL
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "amount": 42,
        "fits": None,
        "maximum": None,
        "model": DEFAULT_MODEL,
        "unit": "model_tokens",
    }


def test_cli_measure_rejects_ambiguous_or_list_mode() -> None:
    with pytest.raises(SystemExit, match="not both"):
        cli.main(["positional text", "--measure", "flag text"])
    with pytest.raises(SystemExit, match="cannot be combined"):
        cli.main(["--list-models", "--measure", "text"])
