from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import inflectsynth._onnxvoice as boundary
from inflectsynth.errors import OnnxVoiceContractError


class FakeInstallation:
    def __init__(self, *, system: str = "inflect", roles: tuple[str, ...] = ("duration", "decode")):
        self.system = system
        self.id = "nano-v2"
        self.sample_rate = None
        self.metadata = {
            "source_revision": "src-rev-1",
            "controls": {"variation": {"default": 0.667}},
            "runtime": {"profile": "inflect-v2-split-v1"},
            "voice_details": {"default": {"id": "default", "locale": "en-US"}},
        }
        self._artifacts = {role: SimpleNamespace(path=Path(f"/{role}.onnx")) for role in roles}

    @property
    def ref(self) -> str:
        return f"{self.system}:{self.id}"

    def require_artifact(self, role: str) -> Any:
        try:
            return self._artifacts[role]
        except KeyError:
            raise KeyError(role) from None


class FakeManager:
    instances: list[FakeManager] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, Any]] = []
        self.installation = FakeInstallation()
        self.__class__.instances.append(self)

    def install(self, ref: str, **kwargs: Any) -> FakeInstallation:
        self.calls.append(("install", (ref, kwargs)))
        return self.installation

    def open(self, installation: Any, **kwargs: Any) -> object:
        self.calls.append(("open", (installation, kwargs)))
        return SimpleNamespace(installation=installation)

    def list(self, system: str, **kwargs: Any) -> list[object]:
        self.calls.append(("list", (system, kwargs)))
        return [SimpleNamespace(id="nano-v2")]

    def list_voices(self, system: str, **kwargs: Any) -> list[object]:
        self.calls.append(("list_voices", (system, kwargs)))
        return [SimpleNamespace(id="default")]


class FakeOnnxVoice:
    open_local_calls: list[dict[str, Any]] = []

    def __new__(cls, **kwargs: Any) -> FakeManager:
        return FakeManager(**kwargs)

    @staticmethod
    def open_local(**kwargs: Any) -> object:
        FakeOnnxVoice.open_local_calls.append(kwargs)
        return SimpleNamespace(installation=FakeInstallation())


def _fake_module() -> SimpleNamespace:
    return SimpleNamespace(
        OnnxVoice=FakeOnnxVoice,
        registered_systems=lambda: ("inflect", "piper"),
    )


def test_normalize_ref_adds_inflect_system_and_rejects_empty() -> None:
    assert boundary.normalize_ref("nano-v2") == "inflect:nano-v2"
    assert boundary.normalize_ref("inflect:nano-v2") == "inflect:nano-v2"
    with pytest.raises(OnnxVoiceContractError, match="must not be empty"):
        boundary.normalize_ref(" ")


def test_install_delegates_refresh_force_progress_and_catalog_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeManager.instances.clear()
    monkeypatch.setattr(boundary, "_module", _fake_module)
    progress = object()

    model = boundary.install_pretrained_model(
        "nano-v2",
        cache_dir="/cache",
        offline=True,
        refresh_catalog=True,
        force_download=True,
        catalog_url="file:///catalog.json",
        progress=progress,
    )

    manager = FakeManager.instances[-1]
    assert manager.kwargs == {
        "cache_dir": "/cache",
        "catalog_sources": {"inflect": "file:///catalog.json"},
        "offline": True,
    }
    assert manager.calls == [
        (
            "install",
            (
                "inflect:nano-v2",
                {"refresh": True, "force": True, "progress": progress},
            ),
        )
    ]
    assert model.ref == "inflect:nano-v2"
    assert model.duration_path == Path("/duration.onnx")
    assert model.decode_path == Path("/decode.onnx")
    assert model.sample_rate == 24_000
    assert model.metadata["source_revision"] == "src-rev-1"
    assert model.metadata["controls"]["variation"]["default"] == 0.667
    assert model.metadata["voices"]["default"]["locale"] == "en-US"


def test_managed_open_forwards_provider_and_session_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeManager.instances.clear()
    monkeypatch.setattr(boundary, "_module", _fake_module)
    installation = FakeInstallation()
    model = boundary.installation_to_model(installation)
    provider_options = [{"device_id": "2"}]
    session_options = object()

    runtime = boundary.open_installed_model(
        model,
        providers=["CPUExecutionProvider"],
        provider_options=provider_options,
        session_options=session_options,
    )

    _, (opened_installation, options) = FakeManager.instances[-1].calls[-1]
    assert opened_installation is installation
    assert options == {
        "providers": ["CPUExecutionProvider"],
        "provider_options": provider_options,
        "session_options": session_options,
    }
    assert runtime.installation is installation


def test_local_open_uses_duration_and_decode_artifact_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeOnnxVoice.open_local_calls.clear()
    monkeypatch.setattr(boundary, "_module", _fake_module)
    provider_options = {"CPUExecutionProvider": {"intra_op_num_threads": 1}}
    session_options = object()

    boundary.open_local_model(
        duration_path="duration.onnx",
        decode_path="decode.onnx",
        sample_rate=24_000,
        metadata={"id": "local"},
        providers="cpu",
        provider_options=provider_options,
        session_options=session_options,
    )

    assert FakeOnnxVoice.open_local_calls == [
        {
            "system": "inflect",
            "artifacts": {"duration": "duration.onnx", "decode": "decode.onnx"},
            "sample_rate": 24_000,
            "metadata": {"id": "local"},
            "providers": "cpu",
            "provider_options": provider_options,
            "session_options": session_options,
        }
    ]


def test_installation_requires_inflect_system_and_both_artifact_roles() -> None:
    with pytest.raises(OnnxVoiceContractError, match="system must be 'inflect'"):
        boundary.installation_to_model(FakeInstallation(system="piper"))
    with pytest.raises(OnnxVoiceContractError, match="duration and decode"):
        boundary.installation_to_model(FakeInstallation(roles=("duration",)))


def test_missing_inflect_adapter_is_a_stable_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        boundary.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            OnnxVoice=FakeOnnxVoice,
            registered_systems=lambda: ("piper",),
        ),
    )
    message = "does not expose the required Inflect adapter"
    with pytest.raises(OnnxVoiceContractError, match=message):
        boundary._module()


def test_catalog_helpers_only_list_models_and_voices(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeManager.instances.clear()
    monkeypatch.setattr(boundary, "_module", _fake_module)

    assert boundary.catalog_items(language="en-US", offline=True, refresh=True)
    assert boundary.catalog_voices(language="en-US", offline=True, refresh=True)
    calls = [call[0] for manager in FakeManager.instances for call in manager.calls]
    assert calls == ["list", "list_voices"]


def test_onnxvoice_errors_are_translated_to_stable_public_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOnnxVoiceError(Exception):
        pass

    class FakeOfflineError(FakeOnnxVoiceError):
        pass

    class FakeAssetNotFoundError(FakeOnnxVoiceError):
        pass

    class FakeRuntimeContractError(FakeOnnxVoiceError):
        pass

    errors = SimpleNamespace(
        OnnxVoiceError=FakeOnnxVoiceError,
        OfflineError=FakeOfflineError,
        AssetNotFoundError=FakeAssetNotFoundError,
        RuntimeContractError=FakeRuntimeContractError,
    )
    monkeypatch.setattr(boundary.importlib, "import_module", lambda _name: errors)

    from inflectsynth.errors import (
        CatalogUnavailableError,
        OnnxVoiceContractError,
        UnsupportedModelError,
    )

    assert isinstance(
        boundary._translate_onnxvoice_error(FakeOfflineError(), "catalog"),
        CatalogUnavailableError,
    )
    assert isinstance(
        boundary._translate_onnxvoice_error(FakeAssetNotFoundError(), "install"),
        UnsupportedModelError,
    )
    assert isinstance(
        boundary._translate_onnxvoice_error(FakeRuntimeContractError(), "open"),
        OnnxVoiceContractError,
    )
