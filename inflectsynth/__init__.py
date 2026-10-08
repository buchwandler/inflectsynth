"""Inflect v2 prepared-text synthesis backed by ONNXVoice and InflectG2P."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from .api_contract import (
    CAPACITY_API_VERSION,
    REQUEST_API_VERSION,
    capacity_api_contract,
    request_api_contract,
)
from .catalog import list_models
from .config import SynthesisConfig
from .constants import DEFAULT_MODEL, DEFAULT_VOICE, SAMPLE_RATE
from .discovery import DescribedVoice, DiscoveredModel, discover_models
from .errors import (
    ArtifactIntegrityError,
    CatalogDiscoveryError,
    CatalogError,
    CatalogUnavailableError,
    EmptyTextError,
    InflectSynthError,
    InvalidSeedError,
    InvalidSpeedError,
    InvalidSynthesisConfigError,
    InvalidVariationError,
    InvalidVoiceError,
    ModelInferenceError,
    ModelNotFoundError,
    OfflineModelError,
    OnnxVoiceContractError,
    ProviderUnavailableError,
    SynthesisInputTooLongError,
    TextPreparationError,
    UnsupportedModelError,
)
from .identity import runtime_identity
from .types import ModelInfo, RequestMeasure, SynthesisResult, VoiceInfo
from .voice import InflectVoice
from .voice_level import (
    CalibrationDataError,
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelApplication,
    VoiceLevelCalibration,
    VoiceLevelConfig,
    VoiceLevelMode,
    apply_voice_level_calibration,
    default_voice_calibration,
    load_voice_calibration,
)

try:
    __version__ = _distribution_version("inflectsynth")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "REQUEST_API_VERSION",
    "request_api_contract",
    "CAPACITY_API_VERSION",
    "capacity_api_contract",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "SAMPLE_RATE",
    "runtime_identity",
    "discover_models",
    "DescribedVoice",
    "DiscoveredModel",
    "InflectVoice",
    "SynthesisConfig",
    "SynthesisResult",
    "RequestMeasure",
    "VoiceInfo",
    "VoiceLevelConfig",
    "VoiceLevelMode",
    "VoiceCalibrationKey",
    "VoiceLevelCalibration",
    "VoiceCalibrationCatalog",
    "VoiceLevelApplication",
    "CalibrationDataError",
    "apply_voice_level_calibration",
    "default_voice_calibration",
    "load_voice_calibration",
    "InflectSynthError",
    "EmptyTextError",
    "TextPreparationError",
    "SynthesisInputTooLongError",
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
    # Transitional prototype aliases, not part of the v0.1.0 request contract.
    "list_models",
    "ModelInfo",
    "CatalogError",
    "ModelNotFoundError",
    "ArtifactIntegrityError",
    "OfflineModelError",
    "ProviderUnavailableError",
]
