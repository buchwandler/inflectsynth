from __future__ import annotations

import sys
from types import MappingProxyType

import pytest

import inflectsynth.identity as identity


def test_runtime_identity_uses_package_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "inflectsynth": "0.1.0",
        "inflectg2p": "0.1.0",
        "onnxvoice": "0.2.5",
    }
    monkeypatch.setattr(identity, "_distribution_version", versions.__getitem__)
    result = identity.runtime_identity()
    assert isinstance(result, MappingProxyType)
    assert result == {
        "engine": "inflect",
        "engine_version": "0.1.0",
        "g2p_revision": "0.1.0",
        "runtime_revision": "0.2.5",
        "request_api_version": "1",
    }
    with pytest.raises(TypeError):
        result["engine"] = "changed"  # type: ignore[index]
    assert "onnxvoice" not in sys.modules
    assert "onnxruntime" not in sys.modules


def test_missing_distribution_metadata_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_distribution: str) -> str:
        raise identity.PackageNotFoundError

    monkeypatch.setattr(identity, "_distribution_version", missing)
    result = identity.runtime_identity()
    assert result["engine_version"] is None
    assert result["g2p_revision"] is None
    assert result["runtime_revision"] is None
