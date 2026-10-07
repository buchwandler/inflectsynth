from __future__ import annotations

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
    assert callable(inflectsynth.InflectVoice.close)
