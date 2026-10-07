from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import inflectsynth.voice as voice_module
from inflectsynth import InflectVoice
from inflectsynth._onnxvoice import ResolvedInflectModel


class FakeG2P:
    def close(self) -> None:
        pass


class FakeRuntime:
    def __init__(self, installation: Any = None) -> None:
        self.installation = installation
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _managed_model() -> ResolvedInflectModel:
    return ResolvedInflectModel(
        ref="inflect:nano-v2",
        model_id="nano-v2",
        duration_path=Path("duration.onnx"),
        decode_path=Path("decode.onnx"),
        sample_rate=24_000,
        metadata={"voices": {"default": {"id": "default"}}},
        installation=object(),
    )


def test_from_pretrained_routes_install_and_runtime_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = _managed_model()
    runtime = FakeRuntime()
    install_calls: list[tuple[str, dict[str, Any]]] = []
    open_calls: list[tuple[ResolvedInflectModel, dict[str, Any]]] = []

    def install(model: str, **kwargs: Any) -> ResolvedInflectModel:
        install_calls.append((model, kwargs))
        return installed

    def open_model(model: ResolvedInflectModel, **kwargs: Any) -> FakeRuntime:
        open_calls.append((model, kwargs))
        return runtime

    monkeypatch.setattr(voice_module, "install_pretrained_model", install)
    monkeypatch.setattr(voice_module, "open_installed_model", open_model)
    g2p = FakeG2P()
    progress = object()
    provider_options = {"CPUExecutionProvider": {"intra_op_num_threads": 2}}
    session_options = object()

    voice = InflectVoice.from_pretrained(
        "nano",
        cache_dir="/models",
        offline=True,
        refresh_catalog=True,
        force_download=True,
        catalog_url="/catalog.json",
        providers="cpu",
        provider_options=provider_options,
        session_options=session_options,
        progress=progress,
        g2p=g2p,  # type: ignore[arg-type]
    )

    assert install_calls == [
        (
            "nano",
            {
                "cache_dir": "/models",
                "offline": True,
                "refresh_catalog": True,
                "force_download": True,
                "catalog_url": "/catalog.json",
                "progress": progress,
            },
        )
    ]
    assert open_calls == [
        (
            installed,
            {
                "cache_dir": "/models",
                "offline": True,
                "catalog_url": "/catalog.json",
                "providers": "cpu",
                "provider_options": provider_options,
                "session_options": session_options,
            },
        )
    ]
    assert voice.model_ref == "inflect:nano-v2"
    voice.close()
    assert runtime.closed == 1


def test_from_local_uses_onnxvoice_local_open_without_calibration_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installation = SimpleNamespace(
        system="inflect",
        id="local-bundle",
        sample_rate=24_000,
        metadata={
            "voice_details": {"default": {"id": "default", "locale": "en-US"}},
            "default_voice": "default",
        },
        require_artifact=lambda role: SimpleNamespace(path=Path(f"{role}.onnx")),
    )
    runtime = FakeRuntime(installation)
    open_calls: list[dict[str, Any]] = []

    def open_local(**kwargs: Any) -> FakeRuntime:
        open_calls.append(kwargs)
        return runtime

    monkeypatch.setattr(voice_module, "open_local_model", open_local)
    g2p = FakeG2P()
    voice = InflectVoice.from_local(
        duration_path="duration.onnx",
        decode_path="decode.onnx",
        model_id="custom-local",
        providers=["CPUExecutionProvider"],
        provider_options=[{}],
        session_options=object(),
        g2p=g2p,  # type: ignore[arg-type]
    )

    assert open_calls[0]["duration_path"] == "duration.onnx"
    assert open_calls[0]["decode_path"] == "decode.onnx"
    assert open_calls[0]["sample_rate"] == 24_000
    assert open_calls[0]["providers"] == ["CPUExecutionProvider"]
    assert voice.model_id == "custom-local"
    assert voice.model_ref is None
    assert voice.calibration_key("default") is None
    voice.close()
    assert runtime.closed == 1
