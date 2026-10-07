from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import (
    CatalogDiscoveryError,
    CatalogUnavailableError,
    InflectSynthError,
    OnnxVoiceContractError,
    UnsupportedModelError,
)

DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/buchwandler/inflect-onnx-bundles/main/catalog/models.json"
)


@dataclass(frozen=True, slots=True)
class ResolvedInflectModel:
    ref: str | None
    model_id: str
    duration_path: Path
    decode_path: Path
    sample_rate: int
    metadata: dict[str, Any]
    installation: Any


def normalize_ref(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OnnxVoiceContractError("Inflect model reference must not be empty")
    return value if ":" in value else f"inflect:{value}"


def _module() -> ModuleType:
    """Import ONNXVoice only when an operation needs its managed runtime."""
    try:
        module = importlib.import_module("onnxvoice")
    except Exception as exc:
        raise OnnxVoiceContractError(
            "ONNXVoice with the Inflect adapter is required; install a compatible onnxvoice release"
        ) from exc

    manager_type = getattr(module, "OnnxVoice", None)
    systems = getattr(module, "registered_systems", None)
    try:
        available_systems = systems() if callable(systems) else ()
    except Exception as exc:
        raise OnnxVoiceContractError("Could not inspect the ONNXVoice adapter registry") from exc
    if manager_type is None or "inflect" not in available_systems:
        raise OnnxVoiceContractError(
            "Installed ONNXVoice does not expose the required Inflect adapter contract"
        )
    return module


def _manager(
    *,
    cache_dir: str | Path | None,
    offline: bool,
    catalog_url: str | None,
) -> Any:
    module = _module()
    source = catalog_url or os.environ.get("INFLECTSYNTH_CATALOG_URL") or DEFAULT_CATALOG_URL
    try:
        return module.OnnxVoice(
            cache_dir=cache_dir,
            catalog_sources={"inflect": source},
            offline=offline,
        )
    except TypeError as exc:
        raise OnnxVoiceContractError(
            "Installed ONNXVoice does not support the Inflect catalog_sources manager contract"
        ) from exc


def _translate_onnxvoice_error(exc: Exception, operation: str) -> Exception:
    """Translate ONNXVoice's public error families to stable InflectSynth errors."""
    try:
        errors = importlib.import_module("onnxvoice.errors")
    except Exception:
        return OnnxVoiceContractError(f"ONNXVoice {operation} operation failed")

    def is_error(name: str) -> bool:
        error_type = getattr(errors, name, None)
        return isinstance(error_type, type) and isinstance(exc, error_type)

    message = f"ONNXVoice {operation} operation failed: {exc}"
    if is_error("OfflineError"):
        return CatalogUnavailableError(message)
    if is_error("CatalogError"):
        return CatalogDiscoveryError(message)
    if is_error("AssetNotFoundError") or is_error("IntegrityError"):
        return UnsupportedModelError(message)
    if is_error("AssetDownloadError"):
        return UnsupportedModelError(message)
    if is_error("RuntimeContractError") or is_error("OptionalDependencyError"):
        return OnnxVoiceContractError(message)
    if is_error("UnsupportedSystemError"):
        return UnsupportedModelError(message)
    if operation == "catalog":
        return CatalogUnavailableError(message)
    if operation == "install":
        return UnsupportedModelError(message)
    return OnnxVoiceContractError(message)


def installation_to_model(installation: Any) -> ResolvedInflectModel:
    """Validate the public ONNXVoice installation shape required by InflectSynth."""
    if getattr(installation, "system", None) != "inflect":
        raise OnnxVoiceContractError("ONNXVoice installation system must be 'inflect'")
    require_artifact = getattr(installation, "require_artifact", None)
    if not callable(require_artifact):
        raise OnnxVoiceContractError("ONNXVoice installation must support require_artifact(role)")
    try:
        duration = require_artifact("duration")
        decode = require_artifact("decode")
    except (KeyError, TypeError, ValueError) as exc:
        raise OnnxVoiceContractError(
            "ONNXVoice Inflect installation must contain duration and decode artifacts"
        ) from exc

    try:
        duration_path = Path(duration.path)
        decode_path = Path(decode.path)
        metadata_value = installation.metadata
        metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        model_id = installation.id
    except (AttributeError, TypeError, ValueError) as exc:
        raise OnnxVoiceContractError("ONNXVoice returned an invalid Inflect installation") from exc
    if not isinstance(model_id, str) or not model_id:
        raise OnnxVoiceContractError("ONNXVoice Inflect installation must have a model id")

    sample_rate = getattr(installation, "sample_rate", None)
    if sample_rate is None:
        sample_rate = metadata.get("sample_rate")
    if sample_rate is None:
        sample_rate = 24_000
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise OnnxVoiceContractError("ONNXVoice Inflect sample rate must be a positive integer")

    metadata.setdefault("id", model_id)
    metadata.setdefault("sample_rate", sample_rate)
    metadata.setdefault("default_voice", getattr(installation, "default_voice", None) or "default")
    if "voices" not in metadata:
        voice_details = metadata.get("voice_details")
        if isinstance(voice_details, Mapping):
            metadata["voices"] = {key: dict(value) for key, value in voice_details.items()}

    ref_value = getattr(installation, "ref", None)
    ref = ref_value if isinstance(ref_value, str) else normalize_ref(model_id)
    return ResolvedInflectModel(
        ref=ref,
        model_id=model_id,
        duration_path=duration_path,
        decode_path=decode_path,
        sample_rate=sample_rate,
        metadata=metadata,
        installation=installation,
    )


def install_pretrained_model(
    model: str,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    refresh_catalog: bool = False,
    force_download: bool = False,
    catalog_url: str | None = None,
    progress: Any | None = None,
) -> ResolvedInflectModel:
    manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
    ref = normalize_ref(model)
    try:
        installation = manager.install(
            ref,
            refresh=refresh_catalog,
            force=force_download,
            progress=progress,
        )
    except TypeError as exc:
        raise OnnxVoiceContractError(
            "Installed ONNXVoice does not support the Inflect install contract"
        ) from exc
    except InflectSynthError:
        raise
    except Exception as exc:
        raise _translate_onnxvoice_error(exc, "install") from exc
    return installation_to_model(installation)


def open_installed_model(
    model: ResolvedInflectModel,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    catalog_url: str | None = None,
    providers: Sequence[Any] | str | None = None,
    provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
    session_options: Any | None = None,
) -> Any:
    manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
    try:
        return manager.open(
            model.installation,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
    except TypeError as exc:
        raise OnnxVoiceContractError(
            "Installed ONNXVoice does not support provider/session options for Inflect models"
        ) from exc
    except InflectSynthError:
        raise
    except Exception as exc:
        raise _translate_onnxvoice_error(exc, "open") from exc


def open_local_model(
    *,
    duration_path: str | Path,
    decode_path: str | Path,
    sample_rate: int = 24_000,
    metadata: Mapping[str, Any] | None = None,
    providers: Sequence[Any] | str | None = None,
    provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
    session_options: Any | None = None,
) -> Any:
    module = _module()
    try:
        return module.OnnxVoice.open_local(
            system="inflect",
            artifacts={"duration": duration_path, "decode": decode_path},
            sample_rate=sample_rate,
            metadata=dict(metadata or {}),
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
    except TypeError as exc:
        raise OnnxVoiceContractError(
            "Installed ONNXVoice does not support local Inflect split-artifact opening"
        ) from exc
    except InflectSynthError:
        raise
    except Exception as exc:
        raise _translate_onnxvoice_error(exc, "open_local") from exc


def catalog_items(
    *,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
    catalog_url: str | None = None,
) -> tuple[Any, ...]:
    """Return catalog records only; never install or open model artifacts."""
    manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
    try:
        return tuple(manager.list("inflect", language=language, refresh=refresh))
    except TypeError as exc:
        raise OnnxVoiceContractError(
            "Installed ONNXVoice does not support Inflect catalog discovery"
        ) from exc
    except InflectSynthError:
        raise
    except Exception as exc:
        raise _translate_onnxvoice_error(exc, "catalog") from exc


def catalog_voices(
    *,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
    catalog_url: str | None = None,
) -> tuple[Any, ...]:
    manager = _manager(cache_dir=cache_dir, offline=offline, catalog_url=catalog_url)
    try:
        return tuple(manager.list_voices("inflect", language=language, refresh=refresh))
    except TypeError as exc:
        raise OnnxVoiceContractError(
            "Installed ONNXVoice does not support Inflect voice discovery"
        ) from exc
    except InflectSynthError:
        raise
    except Exception as exc:
        raise _translate_onnxvoice_error(exc, "catalog") from exc
