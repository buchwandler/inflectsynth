from __future__ import annotations

import inspect

import inflectsynth


def test_root_exports_the_readio_request_contract() -> None:
    required = (
        "InflectVoice",
        "SynthesisConfig",
        "SynthesisResult",
        "VoiceInfo",
        "discover_models",
        "runtime_identity",
        "REQUEST_API_VERSION",
        "request_api_contract",
        "CAPACITY_API_VERSION",
        "capacity_api_contract",
        "DEFAULT_MODEL",
        "DEFAULT_VOICE",
        "SAMPLE_RATE",
        "RequestMeasure",
        "TextPreparationError",
        "SynthesisInputTooLongError",
        "VoiceLevelConfig",
        "InflectSynthError",
        "EmptyTextError",
        "InvalidSpeedError",
        "InvalidVariationError",
        "InvalidSeedError",
        "InvalidVoiceError",
        "InvalidSynthesisConfigError",
        "UnsupportedModelError",
        "OnnxVoiceContractError",
        "ModelInferenceError",
        "CatalogDiscoveryError",
        "CatalogUnavailableError",
    )
    assert all(hasattr(inflectsynth, name) for name in required)
    assert all(name in inflectsynth.__all__ for name in required)


def test_voice_has_managed_local_atomic_and_lifecycle_methods() -> None:
    assert callable(inflectsynth.InflectVoice.from_pretrained)
    assert callable(inflectsynth.InflectVoice.from_local)
    assert callable(inflectsynth.InflectVoice.synthesize_prepared)
    assert callable(inflectsynth.InflectVoice.measure_prepared)
    assert callable(inflectsynth.InflectVoice.close)


def test_public_defaults_match_the_exported_constants() -> None:
    assert inflectsynth.DEFAULT_MODEL == "nano-v2"
    assert inflectsynth.DEFAULT_VOICE == "default"
    assert inflectsynth.SAMPLE_RATE == 24_000
    assert (
        inspect.signature(inflectsynth.InflectVoice.from_pretrained).parameters["model"].default
        == inflectsynth.DEFAULT_MODEL
    )
    assert (
        inspect.signature(inflectsynth.InflectVoice.synthesize_prepared).parameters["voice"].default
        == inflectsynth.DEFAULT_VOICE
    )
