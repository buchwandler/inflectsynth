"""Inflect v2 ONNX synthesis frontend."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .catalog import list_models, load_catalog, resolve_model
from .errors import (
    ArtifactIntegrityError,
    CatalogError,
    InflectSynthError,
    InvalidSynthesisConfigError,
    InvalidVoiceError,
    ModelNotFoundError,
    OfflineModelError,
    ProviderUnavailableError,
)
from .install import install_model
from .types import InstalledModel, ModelInfo, SynthesisResult, VoiceInfo
from .voice import InflectVoice, boundary_pause_seconds, edge_fade, split_text

try:
    __version__ = _distribution_version("inflectsynth")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "ArtifactIntegrityError",
    "CatalogError",
    "InflectSynthError",
    "InflectVoice",
    "InstalledModel",
    "InvalidSynthesisConfigError",
    "InvalidVoiceError",
    "ModelInfo",
    "ModelNotFoundError",
    "OfflineModelError",
    "ProviderUnavailableError",
    "SynthesisResult",
    "VoiceInfo",
    "boundary_pause_seconds",
    "edge_fade",
    "install_model",
    "list_models",
    "load_catalog",
    "resolve_model",
    "split_text",
]
