from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import inflectsynth
from inflectsynth import InflectVoice, RequestMeasure, SynthesisInputTooLongError
from inflectsynth._onnxvoice import ResolvedInflectModel
from inflectsynth.errors import EmptyTextError, TextPreparationError


class FakeG2P:
    def __init__(self, token_ids: tuple[int, ...] = (4, 5, 6)) -> None:
        self.token_ids = token_ids
        self.calls: list[str] = []
        self.error: Exception | None = None
        self.closed = 0

    def phonemize_prepared(self, text: str) -> Any:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=text.strip(), phonemes="fəˈnetɪks", token_ids=self.token_ids)

    def close(self) -> None:
        self.closed += 1


class FakeRuntime:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.closed = 0

    def infer(self, token_ids: tuple[int, ...], **_kwargs: Any) -> Any:
        self.calls.append(token_ids)
        return SimpleNamespace(audio=(0.1, -0.1), sample_rate=self.sample_rate)

    def close(self) -> None:
        self.closed += 1


def make_voice(
    *,
    managed: bool = True,
    token_ids: tuple[int, ...] = (4, 5, 6),
    model_id_override: str | None = None,
) -> tuple[InflectVoice, FakeRuntime, FakeG2P]:
    runtime = FakeRuntime()
    g2p = FakeG2P(token_ids)
    model_id = model_id_override or ("nano-v2" if managed else "local-model")
    installed = ResolvedInflectModel(
        ref=f"inflect:{model_id}" if managed else None,
        model_id=model_id,
        duration_path=Path("duration.onnx"),
        decode_path=Path("decode.onnx"),
        sample_rate=24_000,
        metadata={
            "id": model_id,
            "voices": {"default": {"id": "default"}},
        },
        installation=object(),
    )
    return InflectVoice(runtime=runtime, installed=installed, g2p=g2p), runtime, g2p


def test_measurement_uses_exact_input_and_never_runs_acoustic_inference() -> None:
    text = "  Keep   this prepared text.  "
    voice, runtime, g2p = make_voice()
    previous_application = voice.last_voice_level_application

    measure = voice.measure_prepared(text)

    assert g2p.calls == [text]
    assert measure == RequestMeasure(fits=True, amount=3, maximum=400, model_id="nano-v2")
    assert measure.unit == "model_tokens"
    assert measure.token_count == measure.amount == 3
    assert measure.max_tokens == measure.maximum == 400
    assert runtime.calls == []
    assert voice.last_voice_level_application is previous_application


def test_measurement_and_synthesis_report_the_same_frontend_token_count() -> None:
    voice, runtime, g2p = make_voice(token_ids=(12, 9, 2, 7))
    text = "A prepared request."

    measure = voice.measure_prepared(text)
    result = voice.synthesize_prepared(text)

    assert g2p.calls == [text, text]
    assert measure.amount == result.metadata["token_count"] == 4
    assert runtime.calls == [(12, 9, 2, 7)]


def test_managed_capacity_matches_policy_and_local_models_stay_unknown() -> None:
    managed, _, _ = make_voice(token_ids=(1, 2, 3))
    measurement = managed.measure_prepared("fits")
    assert measurement.maximum == 400
    assert measurement.fits is True

    local, _, _ = make_voice(managed=False)
    local_measurement = local.measure_prepared("unknown")
    assert local_measurement.maximum is None
    assert local_measurement.fits is None

    local_spoof, _, _ = make_voice(managed=False, model_id_override="nano-v2")
    spoof_measurement = local_spoof.measure_prepared("must remain unknown")
    assert spoof_measurement.maximum is None
    assert spoof_measurement.fits is None


def test_exact_policy_limit_fits_and_over_limit_fails_before_inference() -> None:
    exact, exact_runtime, _ = make_voice(token_ids=tuple(range(400)))
    exact_measure = exact.measure_prepared("exactly at limit")
    assert exact_measure.amount == exact_measure.maximum == 400
    assert exact_measure.fits is True
    result = exact.synthesize_prepared("exactly at limit")
    assert result.metadata["token_count"] == exact_measure.amount
    assert exact_runtime.calls == [tuple(range(400))]

    over, over_runtime, _ = make_voice(token_ids=tuple(range(401)))
    over_measure = over.measure_prepared("over limit")
    assert over_measure.amount == 401
    assert over_measure.maximum == 400
    assert over_measure.fits is False
    with pytest.raises(SynthesisInputTooLongError) as caught:
        over.synthesize_prepared("over limit")
    assert caught.value.token_count == 401
    assert caught.value.max_tokens == 400
    assert over_runtime.calls == []


def test_stale_managed_model_revision_does_not_inherit_policy_capacity() -> None:
    voice, _, _ = make_voice()
    voice.metadata["source_revision"] = "stale-model-revision"

    measure = voice.measure_prepared("unknown due to stale policy provenance")

    assert measure.maximum is None
    assert measure.fits is None


def test_synthesis_rejects_over_budget_before_inference_with_stable_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voice, runtime, _ = make_voice(token_ids=(1, 2, 3, 4))
    monkeypatch.setattr(voice, "_maximum_input_tokens", lambda: 3)
    text = "too long"

    with pytest.raises(SynthesisInputTooLongError) as caught:
        voice.synthesize_prepared(text)

    error = caught.value
    assert error.token_count == 4
    assert error.max_tokens == 3
    assert error.text_length == len(text)
    assert error.model_id == "nano-v2"
    assert runtime.calls == []


def test_request_at_known_capacity_synthesizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voice, runtime, _ = make_voice(token_ids=(1, 2, 3))
    monkeypatch.setattr(voice, "_maximum_input_tokens", lambda: 3)

    result = voice.synthesize_prepared("at limit")

    assert result.metadata["token_count"] == 3
    assert runtime.calls == [(1, 2, 3)]


def test_unknown_capacity_does_not_create_a_synthetic_synthesis_limit() -> None:
    voice, runtime, _ = make_voice(managed=False, token_ids=tuple(range(500)))

    result = voice.synthesize_prepared("long but not policy-limited")

    assert result.metadata["token_count"] == 500
    assert len(runtime.calls) == 1


def test_empty_and_closed_voice_measurement_errors_match_synthesis() -> None:
    voice, _, g2p = make_voice()
    with pytest.raises(EmptyTextError):
        voice.measure_prepared("   ")
    assert g2p.calls == []

    voice.close()
    with pytest.raises(RuntimeError, match="closed"):
        voice.measure_prepared("text")


def test_unexpected_g2p_failure_maps_to_text_preparation_error() -> None:
    voice, runtime, g2p = make_voice()
    g2p.error = RuntimeError("frontend detail")

    with pytest.raises(TextPreparationError) as caught:
        voice.measure_prepared("text")

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert runtime.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fits": None, "amount": -1, "maximum": None},
        {"fits": None, "amount": True, "maximum": None},
        {"fits": None, "amount": 0, "maximum": 0},
        {"fits": None, "amount": 0, "maximum": True},
        {"fits": 1, "amount": 0, "maximum": None},
        {"fits": True, "amount": 2, "maximum": None},
        {"fits": False, "amount": 2, "maximum": 2},
    ],
)
def test_request_measure_rejects_invalid_or_contradictory_values(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        RequestMeasure(**kwargs)


def test_public_capacity_symbols_are_exported() -> None:
    for symbol in (
        "DEFAULT_MODEL",
        "DEFAULT_VOICE",
        "SAMPLE_RATE",
        "RequestMeasure",
        "RequestCapacity",
        "RequestCapacityPolicy",
        "default_request_capacity_policy",
        "load_request_capacity_policy",
        "request_capacity_for",
        "CAPACITY_API_VERSION",
        "capacity_api_contract",
        "TextPreparationError",
        "SynthesisInputTooLongError",
    ):
        assert symbol in inflectsynth.__all__
        assert hasattr(inflectsynth, symbol)
