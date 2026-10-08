"""Strict static voice-level calibration for managed Inflect models."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import audiosig
import numpy as np

from .errors import InflectSynthError, InvalidSynthesisConfigError


class _ReadableTextResource(Protocol):
    """Minimal resource interface needed by the calibration loader."""

    def read_text(self, encoding: str | None = None) -> str: ...


_SUPPORTED_SCHEMA = 1
_SUPPORTED_METHOD = "bs1770"
_RECORD_FIELDS = {
    "gain_db",
    "measured_lufs",
    "reference_lufs",
    "mad_lu",
    "samples",
    "method",
    "corpus_version",
}
_TOP_FIELDS = {"schema", "method", "corpus", "reference_lufs", "generated_with", "voices"}

VoiceLevelMode = Literal["off", "calibrated"]
VoiceLevelSource = Literal[
    "off",
    "override",
    "catalog",
    "missing_identity",
    "missing_calibration",
]


class CalibrationDataError(InflectSynthError, ValueError):
    """Raised when a calibration catalog or identity is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class VoiceLevelConfig:
    """Select opt-in catalog calibration or a caller-provided static gain."""

    mode: VoiceLevelMode = "off"
    gain_db: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"off", "calibrated"}:
            raise InvalidSynthesisConfigError("voice-level mode must be 'off' or 'calibrated'")
        if self.gain_db is not None:
            if isinstance(self.gain_db, bool) or not isinstance(self.gain_db, (int, float)):
                raise InvalidSynthesisConfigError("gain_db must be a finite number or None")
            gain_db = float(self.gain_db)
            if not math.isfinite(gain_db):
                raise InvalidSynthesisConfigError("gain_db must be finite")
            object.__setattr__(self, "gain_db", gain_db)


@dataclass(frozen=True, slots=True, order=True)
class VoiceCalibrationKey:
    """Stable three-part identity for a managed Inflect model and voice."""

    model_source: str
    model_id: str
    voice: str

    def __post_init__(self) -> None:
        values = (self.model_source, self.model_id, self.voice)
        if any(not isinstance(value, str) or not value for value in values):
            raise CalibrationDataError("calibration key components must be non-empty strings")
        if self.model_source != "inflect":
            raise CalibrationDataError("calibration key model_source must be 'inflect'")
        if any(":" in value for value in values):
            raise CalibrationDataError("calibration key components cannot contain ':'")

    def __str__(self) -> str:
        return f"{self.model_source}:{self.model_id}:{self.voice}"

    @classmethod
    def parse(cls, value: str) -> VoiceCalibrationKey:
        if not isinstance(value, str):
            raise CalibrationDataError("calibration key must be a string")
        parts = value.split(":")
        if len(parts) != 3:
            raise CalibrationDataError(
                "calibration key must have inflect:<model-id>:<voice-id> form"
            )
        return cls(*parts)


@dataclass(frozen=True, slots=True)
class VoiceLevelCalibration:
    gain_db: float
    measured_lufs: float | None = None
    reference_lufs: float | None = None
    mad_lu: float | None = None
    samples: int | None = None
    method: str = _SUPPORTED_METHOD
    corpus_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gain_db", _finite(self.gain_db, "gain_db"))
        for field_name in ("measured_lufs", "reference_lufs", "mad_lu"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _finite(value, field_name))
        if self.samples is not None and (
            isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples <= 0
        ):
            raise CalibrationDataError("samples must be a positive integer or None")
        if self.method != _SUPPORTED_METHOD:
            raise CalibrationDataError(f"unsupported calibration method: {self.method!r}")
        if self.corpus_version is not None and not isinstance(self.corpus_version, str):
            raise CalibrationDataError("corpus_version must be a string or None")


@dataclass(frozen=True, slots=True)
class VoiceCalibrationCatalog:
    schema: int
    method: str
    corpus: str
    reference_lufs: float
    generated_with: Mapping[str, str]
    voices: Mapping[VoiceCalibrationKey, VoiceLevelCalibration]
    revision: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.schema, bool) or self.schema != _SUPPORTED_SCHEMA:
            raise CalibrationDataError(f"unsupported calibration schema: {self.schema!r}")
        if self.method != _SUPPORTED_METHOD:
            raise CalibrationDataError(f"unsupported calibration method: {self.method!r}")
        if not isinstance(self.corpus, str) or not self.corpus.strip():
            raise CalibrationDataError("corpus must be a non-empty string")
        if not isinstance(self.generated_with, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.generated_with.items()
        ):
            raise CalibrationDataError("generated_with must be a string mapping")
        if not isinstance(self.voices, Mapping) or any(
            not isinstance(key, VoiceCalibrationKey) or not isinstance(value, VoiceLevelCalibration)
            for key, value in self.voices.items()
        ):
            raise CalibrationDataError("voices must map calibration keys to calibration records")
        if self.revision is not None and (
            not isinstance(self.revision, str) or len(self.revision) != 64
        ):
            raise CalibrationDataError("revision must be a SHA-256 hex digest or None")
        object.__setattr__(self, "reference_lufs", _finite(self.reference_lufs, "reference_lufs"))
        object.__setattr__(
            self,
            "generated_with",
            MappingProxyType(dict(sorted(self.generated_with.items()))),
        )
        object.__setattr__(
            self,
            "voices",
            MappingProxyType(dict(sorted(self.voices.items(), key=lambda item: str(item[0])))),
        )


@dataclass(frozen=True, slots=True)
class VoiceLevelApplication:
    applied: bool
    gain_db: float
    source: VoiceLevelSource
    key: VoiceCalibrationKey | None
    mode: VoiceLevelMode = "off"
    catalog_revision: str | None = None
    reason: str = ""

    @property
    def calibration_key(self) -> VoiceCalibrationKey | None:
        return self.key


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationDataError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        raise CalibrationDataError(f"{name} must be finite") from None
    if not math.isfinite(converted):
        raise CalibrationDataError(f"{name} must be finite")
    return converted


def _optional_finite(value: Any, name: str) -> float | None:
    return None if value is None else _finite(value, name)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationDataError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CalibrationDataError(f"non-finite JSON number is not allowed: {value}")


def _validate_record(key: VoiceCalibrationKey, raw: Any) -> VoiceLevelCalibration:
    if not isinstance(raw, Mapping):
        raise CalibrationDataError(f"record for {key} must be an object")
    unknown = set(raw) - _RECORD_FIELDS
    if unknown:
        raise CalibrationDataError(f"record for {key} has unknown field(s): {sorted(unknown)}")
    if "gain_db" not in raw:
        raise CalibrationDataError(f"record for {key} is missing gain_db")
    method = raw.get("method", _SUPPORTED_METHOD)
    if method != _SUPPORTED_METHOD:
        raise CalibrationDataError(f"record for {key} has unsupported method: {method!r}")
    samples = raw.get("samples")
    if samples is not None and (
        isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0
    ):
        raise CalibrationDataError(f"record for {key} samples must be a positive integer")
    corpus_version = raw.get("corpus_version")
    if corpus_version is not None and not isinstance(corpus_version, str):
        raise CalibrationDataError(f"record for {key} corpus_version must be a string or null")
    return VoiceLevelCalibration(
        gain_db=_finite(raw["gain_db"], f"{key}.gain_db"),
        measured_lufs=_optional_finite(raw.get("measured_lufs"), f"{key}.measured_lufs"),
        reference_lufs=_optional_finite(raw.get("reference_lufs"), f"{key}.reference_lufs"),
        mad_lu=_optional_finite(raw.get("mad_lu"), f"{key}.mad_lu"),
        samples=samples,
        method=method,
        corpus_version=corpus_version,
    )


def _validate_catalog(raw: Any) -> VoiceCalibrationCatalog:
    if not isinstance(raw, Mapping):
        raise CalibrationDataError("calibration catalog must be an object")
    unknown = set(raw) - _TOP_FIELDS
    if unknown:
        raise CalibrationDataError(f"calibration catalog has unknown field(s): {sorted(unknown)}")
    schema = raw.get("schema")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema != _SUPPORTED_SCHEMA:
        raise CalibrationDataError(f"unsupported calibration schema: {schema!r}")
    if raw.get("method") != _SUPPORTED_METHOD:
        raise CalibrationDataError(f"unsupported calibration method: {raw.get('method')!r}")
    corpus = raw.get("corpus")
    if not isinstance(corpus, str) or not corpus.strip():
        raise CalibrationDataError("corpus must be a non-empty string")
    generated_with = raw.get("generated_with")
    if not isinstance(generated_with, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in generated_with.items()
    ):
        raise CalibrationDataError("generated_with must be a string mapping")
    voices_raw = raw.get("voices")
    if not isinstance(voices_raw, Mapping):
        raise CalibrationDataError("voices must be an object")
    voices: dict[VoiceCalibrationKey, VoiceLevelCalibration] = {}
    for raw_key, record in voices_raw.items():
        key = VoiceCalibrationKey.parse(raw_key)
        if key in voices:
            raise CalibrationDataError(f"duplicate normalized calibration key: {key}")
        voices[key] = _validate_record(key, record)
    try:
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise CalibrationDataError("calibration catalog is not valid JSON data") from exc
    revision = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return VoiceCalibrationCatalog(
        schema=_SUPPORTED_SCHEMA,
        method=_SUPPORTED_METHOD,
        corpus=corpus,
        reference_lufs=_finite(raw.get("reference_lufs"), "reference_lufs"),
        generated_with=generated_with,
        voices=voices,
        revision=revision,
    )


def _parse_catalog(text: str, source: str) -> VoiceCalibrationCatalog:
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise CalibrationDataError(f"invalid calibration JSON: {source}") from exc
    return _validate_catalog(raw)


def load_voice_calibration(path: Path | str | _ReadableTextResource) -> VoiceCalibrationCatalog:
    """Load and strictly validate a JSON calibration catalog."""
    if isinstance(path, (str, Path)):
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        source_name = str(source)
    else:
        text = path.read_text(encoding="utf-8")
        source_name = str(path)
    return _parse_catalog(text, source_name)


@lru_cache(maxsize=1)
def default_voice_calibration() -> VoiceCalibrationCatalog:
    """Load the packaged calibration catalog once per process."""
    resource = files("inflectsynth").joinpath("data").joinpath("voice_level_calibration.json")
    return _parse_catalog(resource.read_text(encoding="utf-8"), str(resource))


def apply_voice_level_calibration(
    audio: np.ndarray,
    config: VoiceLevelConfig,
    key: VoiceCalibrationKey | None,
    *,
    catalog: VoiceCalibrationCatalog | None = None,
) -> tuple[np.ndarray, VoiceLevelApplication]:
    """Apply one static gain without clipping or measuring runtime loudness."""
    source_audio = np.asarray(audio, dtype=np.float32)
    revision = None
    if config.gain_db is not None:
        gain = float(config.gain_db)
        source: VoiceLevelSource = "override"
    elif config.mode == "off":
        gain = 0.0
        source = "off"
    elif key is None:
        gain = 0.0
        source = "missing_identity"
    else:
        selected = catalog if catalog is not None else default_voice_calibration()
        revision = selected.revision
        calibration = selected.voices.get(key)
        if calibration is None:
            gain = 0.0
            source = "missing_calibration"
        else:
            gain = calibration.gain_db
            source = "catalog"

    result = (
        np.asarray(audiosig.apply_gain_db(source_audio.copy(), gain, clip=False), dtype=np.float32)
        if gain
        else source_audio.copy()
    )
    applied = bool(gain and not np.array_equal(result, source_audio))
    reasons = {
        "off": "voice-level calibration is disabled",
        "override": "an explicit gain_db override was selected",
        "catalog": "a matching calibration catalog entry was selected",
        "missing_identity": "the voice has no stable managed calibration identity",
        "missing_calibration": "no catalog entry matches this managed model and voice",
    }
    application = VoiceLevelApplication(
        applied=applied,
        gain_db=gain,
        source=source,
        key=key,
        mode=config.mode,
        catalog_revision=revision,
        reason=reasons[source],
    )
    return result, application


__all__ = [
    "CalibrationDataError",
    "VoiceCalibrationCatalog",
    "VoiceCalibrationKey",
    "VoiceLevelApplication",
    "VoiceLevelCalibration",
    "VoiceLevelConfig",
    "VoiceLevelMode",
    "apply_voice_level_calibration",
    "default_voice_calibration",
    "load_voice_calibration",
]
