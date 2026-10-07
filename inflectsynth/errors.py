class InflectSynthError(Exception):
    """Base exception for InflectSynth."""


class CatalogError(InflectSynthError):
    pass


class ModelNotFoundError(InflectSynthError):
    pass


class ArtifactIntegrityError(InflectSynthError):
    pass


class OfflineModelError(InflectSynthError):
    pass


class ProviderUnavailableError(InflectSynthError):
    pass


class InvalidVoiceError(InflectSynthError):
    pass


class InvalidSynthesisConfigError(InflectSynthError):
    pass
