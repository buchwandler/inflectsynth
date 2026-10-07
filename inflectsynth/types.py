from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    id: str
    name: str
    language: str
    gender: str
    synthetic: bool


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    name: str
    aliases: tuple[str, ...]
    language: str
    sample_rate: int
    default_voice: str
    voices: tuple[VoiceInfo, ...]
    revision: str


@dataclass(frozen=True, slots=True)
class InstalledModel:
    model_id: str
    root: Path
    duration_path: Path
    decode_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    model_id: str
    voice: str
    speed: float
    variation: float
    seed: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio must be finite")
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "audio", audio)
        object.__setattr__(self, "sample_rate", int(self.sample_rate))

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
