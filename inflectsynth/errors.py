class InflectSynthError(Exception):
    """Base exception for InflectSynth failures."""


class EmptyTextError(InflectSynthError, ValueError):
    """The prepared synthesis text is empty or has the wrong type."""


class TextPreparationError(InflectSynthError, RuntimeError):
    """Prepared text could not be converted into model-ready Inflect token IDs."""


class SynthesisInputTooLongError(InflectSynthError, ValueError):
    """Prepared input exceeds the supported InflectSynth request budget."""

    def __init__(
        self,
        message: str,
        *,
        token_count: int,
        max_tokens: int,
        text_length: int,
        model_id: str,
    ) -> None:
        super().__init__(message)
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.text_length = text_length
        self.model_id = model_id


class InvalidSynthesisConfigError(InflectSynthError, ValueError):
    """The synthesis configuration has an invalid shape or value."""


class InvalidSpeedError(InvalidSynthesisConfigError):
    """The requested synthesis speed is invalid."""


class InvalidVariationError(InvalidSynthesisConfigError):
    """The requested decoder variation is invalid."""


class InvalidSeedError(InvalidSynthesisConfigError):
    """The requested deterministic seed is invalid."""


class InvalidVoiceError(InflectSynthError, ValueError):
    """The requested voice is not provided by this fixed-voice model."""


class UnsupportedModelError(InflectSynthError, RuntimeError):
    """The requested model or installation does not satisfy the Inflect contract."""


class OnnxVoiceContractError(InflectSynthError, RuntimeError):
    """The installed ONNXVoice version lacks a required public Inflect API."""


class ModelInferenceError(InflectSynthError, RuntimeError):
    """The ONNXVoice Inflect runtime failed while generating audio."""


class CatalogDiscoveryError(InflectSynthError, RuntimeError):
    """The Inflect model catalog could not be interpreted or queried."""


class CatalogUnavailableError(CatalogDiscoveryError):
    """The Inflect model catalog is unavailable, including while offline."""


# Compatibility names retained for the prototype's pre-release API.
class CatalogError(CatalogDiscoveryError):
    """Deprecated alias for catalog discovery failures."""


class ModelNotFoundError(UnsupportedModelError):
    """Deprecated alias for unsupported model references."""


class ArtifactIntegrityError(InflectSynthError):
    """Deprecated prototype error; artifact integrity belongs to ONNXVoice."""


class OfflineModelError(CatalogUnavailableError):
    """Deprecated alias for unavailable offline model data."""


class ProviderUnavailableError(InflectSynthError):
    """Deprecated prototype error; provider selection belongs to ONNXVoice."""
