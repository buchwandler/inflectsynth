"""Software identity used by Readio for synthesis provenance and cache keys."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from types import MappingProxyType

from .api_contract import REQUEST_API_VERSION


def _distribution_version_safe(distribution: str) -> str | None:
    try:
        return _distribution_version(distribution)
    except PackageNotFoundError:
        return None


def runtime_identity() -> Mapping[str, str | None]:
    """Return stable package versions only; never inspect models or providers."""
    return MappingProxyType(
        {
            "engine": "inflect",
            "engine_version": _distribution_version_safe("inflectsynth"),
            "g2p_revision": _distribution_version_safe("inflectg2p"),
            "runtime_revision": _distribution_version_safe("onnxvoice"),
            "request_api_version": str(REQUEST_API_VERSION),
        }
    )


__all__ = ["runtime_identity"]
