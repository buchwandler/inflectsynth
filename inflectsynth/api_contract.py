"""Versioned, dependency-light declaration of the Readio request contract."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

REQUEST_API_VERSION = 1

_REQUEST_API_CONTRACT: Mapping[str, str | bool] = MappingProxyType(
    {
        "entrypoint": "InflectVoice.synthesize_prepared",
        "supports_linguistic_tokens": False,
        "supports_pronunciation_overrides": False,
        "supports_whole_request_phonemes": False,
        "supports_speakers": False,
        "supports_word_timings": False,
        "supports_voice_level": True,
        "caller_owns_text_boundaries": True,
    }
)


def request_api_contract() -> Mapping[str, str | bool]:
    """Return an immutable declaration without catalog or model access."""
    return _REQUEST_API_CONTRACT


CAPACITY_API_VERSION = 1

_CAPACITY_API_CONTRACT: Mapping[str, str | bool] = MappingProxyType(
    {
        "entrypoint": "InflectVoice.measure_prepared",
        "unit": "model_tokens",
        "supports_known_maximum": False,
        "caller_owns_text_boundaries": True,
    }
)


def capacity_api_contract() -> Mapping[str, str | bool]:
    """Return the immutable capacity API declaration without model access."""
    return _CAPACITY_API_CONTRACT


__all__ = [
    "REQUEST_API_VERSION",
    "request_api_contract",
    "CAPACITY_API_VERSION",
    "capacity_api_contract",
]
