from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

from .errors import (
    InvalidSeedError,
    InvalidSpeedError,
    InvalidSynthesisConfigError,
    InvalidVariationError,
)
from .voice_level import VoiceLevelConfig


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    speed: float = 1.0
    variation: float = 0.667
    seed: int = 0
    voice_level: VoiceLevelConfig = field(default_factory=VoiceLevelConfig)

    def validated(self) -> SynthesisConfig:
        speed = _finite_control(self.speed, "speed", InvalidSpeedError)
        if not 0.5 <= speed <= 2.0:
            raise InvalidSpeedError("speed must be between 0.5 and 2.0")
        variation = _finite_control(self.variation, "variation", InvalidVariationError)
        if not 0.0 <= variation <= 1.0:
            raise InvalidVariationError("variation must be between 0.0 and 1.0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise InvalidSeedError("seed must be a non-negative integer")
        if not isinstance(self.voice_level, VoiceLevelConfig):
            raise InvalidSynthesisConfigError("voice_level must be a VoiceLevelConfig")
        return SynthesisConfig(
            speed=speed,
            variation=variation,
            seed=self.seed,
            voice_level=self.voice_level,
        )


def _finite_control(
    value: object,
    name: str,
    error_type: type[InvalidSynthesisConfigError],
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise error_type(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise error_type(f"{name} must be a finite real number") from None
    if not math.isfinite(number):
        raise error_type(f"{name} must be finite")
    return number
