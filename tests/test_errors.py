from __future__ import annotations

import inflectsynth
from inflectsynth.errors import (
    CatalogDiscoveryError,
    CatalogUnavailableError,
    EmptyTextError,
    InflectSynthError,
    InvalidSeedError,
    InvalidSpeedError,
    InvalidVariationError,
    InvalidVoiceError,
    ModelInferenceError,
    OnnxVoiceContractError,
    UnsupportedModelError,
)


def test_readio_errors_share_the_public_base_and_catalog_hierarchy() -> None:
    error_types = (
        EmptyTextError,
        InvalidSpeedError,
        InvalidVariationError,
        InvalidSeedError,
        InvalidVoiceError,
        UnsupportedModelError,
        OnnxVoiceContractError,
        ModelInferenceError,
        CatalogDiscoveryError,
    )
    assert all(issubclass(error_type, InflectSynthError) for error_type in error_types)
    assert issubclass(CatalogUnavailableError, CatalogDiscoveryError)
    assert all(
        getattr(inflectsynth, error_type.__name__) is error_type for error_type in error_types
    )
