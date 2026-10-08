from __future__ import annotations

import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    id: str
    name: str
    language: str
    gender: str
    synthetic: bool


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Compatibility discovery value; new callers should use DiscoveredModel."""

    id: str
    name: str
    aliases: tuple[str, ...]
    language: str
    sample_rate: int
    default_voice: str
    voices: tuple[VoiceInfo, ...]
    revision: str


@dataclass(frozen=True, slots=True)
class RequestMeasure:
    """Capacity measurement for one exact prepared-text request."""

    fits: bool | None
    amount: int
    maximum: int | None
    unit: Literal["model_tokens"] = "model_tokens"
    source: str = "inflectsynth.frontend"
    model_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.amount) is not int or self.amount < 0:
            raise ValueError("amount must be a non-negative Python integer")
        if self.maximum is not None and (type(self.maximum) is not int or self.maximum <= 0):
            raise ValueError("maximum must be None or a positive Python integer")
        if self.fits is not None and type(self.fits) is not bool:
            raise ValueError("fits must be True, False, or None")
        if self.unit != "model_tokens":
            raise ValueError("unit must be 'model_tokens'")
        if self.maximum is None:
            if self.fits is not None:
                raise ValueError("fits must be None when maximum is unknown")
        elif self.fits != (self.amount <= self.maximum):
            raise ValueError("fits must match amount <= maximum")


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    model_id: str
    voice: str
    speed: float
    variation: float
    seed: int
    model_ref: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be a one-dimensional mono array")
        audio = np.array(audio, dtype=np.float32, order="C", copy=True)
        if audio.size == 0:
            raise ValueError("audio must not be empty")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio must be finite")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral):
            raise ValueError("sample_rate must be a positive integer")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "audio", audio)
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))

    @property
    def duration_seconds(self) -> float:
        return float(self.audio.size) / float(self.sample_rate)

    def save_wav(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(self.audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm.tobytes())
