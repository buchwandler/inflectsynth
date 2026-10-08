from __future__ import annotations

import operator
from collections.abc import Mapping, MutableMapping
from typing import cast

import pytest

import inflectsynth
import inflectsynth._onnxvoice as onnxvoice_boundary
import inflectsynth.voice as voice_module


def test_request_api_contract_is_exact_immutable_and_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_runtime_accessed(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("request contract inspection must not access ONNXVoice")

    monkeypatch.setattr(onnxvoice_boundary, "_module", fail_if_runtime_accessed)
    contract = inflectsynth.request_api_contract()
    assert isinstance(contract, Mapping)
    assert contract == {
        "entrypoint": "InflectVoice.synthesize_prepared",
        "supports_linguistic_tokens": False,
        "supports_pronunciation_overrides": False,
        "supports_whole_request_phonemes": False,
        "supports_speakers": False,
        "supports_word_timings": False,
        "supports_voice_level": True,
        "caller_owns_text_boundaries": True,
    }
    with pytest.raises(TypeError):
        operator.setitem(cast(MutableMapping[str, object], contract), "supports_speakers", True)
    assert inflectsynth.REQUEST_API_VERSION == 1


def test_capacity_api_contract_is_additive_and_dependency_light(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_runtime_accessed(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("capacity contract inspection must not access ONNXVoice")

    monkeypatch.setattr(onnxvoice_boundary, "_module", fail_if_runtime_accessed)
    monkeypatch.setattr(voice_module, "InflectG2P", fail_if_runtime_accessed)
    contract = inflectsynth.capacity_api_contract()
    assert isinstance(contract, Mapping)
    assert contract == {
        "entrypoint": "InflectVoice.measure_prepared",
        "unit": "model_tokens",
        "supports_known_maximum": False,
        "caller_owns_text_boundaries": True,
    }
    with pytest.raises(TypeError):
        operator.setitem(
            cast(MutableMapping[str, object], contract), "supports_known_maximum", True
        )
    assert inflectsynth.CAPACITY_API_VERSION == 1
    assert inflectsynth.REQUEST_API_VERSION == 1
