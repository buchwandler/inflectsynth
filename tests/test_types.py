from __future__ import annotations

import wave

import numpy as np
import pytest

from inflectsynth import SynthesisConfig, SynthesisResult, VoiceLevelConfig
from inflectsynth.errors import (
    InvalidSeedError,
    InvalidSpeedError,
    InvalidSynthesisConfigError,
    InvalidVariationError,
)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 0.49, 2.01])
def test_invalid_speed_is_rejected(value: object) -> None:
    with pytest.raises(InvalidSpeedError):
        SynthesisConfig(speed=value).validated()  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.01, 1.01])
def test_invalid_variation_is_rejected(value: object) -> None:
    with pytest.raises(InvalidVariationError):
        SynthesisConfig(variation=value).validated()  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_invalid_seed_is_rejected(value: object) -> None:
    with pytest.raises(InvalidSeedError):
        SynthesisConfig(seed=value).validated()  # type: ignore[arg-type]


def test_config_requires_typed_voice_level() -> None:
    with pytest.raises(InvalidSynthesisConfigError):
        SynthesisConfig(voice_level=object()).validated()  # type: ignore[arg-type]
    assert SynthesisConfig().validated().voice_level == VoiceLevelConfig()


def test_synthesis_result_owns_audio_and_freezes_metadata(tmp_path) -> None:
    source = np.asarray([0.1, -0.1], dtype=np.float32)
    metadata = {"voice_level": {"mode": "off"}, "items": ["one"]}
    result = SynthesisResult(
        audio=source,
        sample_rate=24_000,
        model_id="nano-v2",
        voice="default",
        speed=1.0,
        variation=0.667,
        seed=5,
        model_ref="inflect:nano-v2",
        metadata=metadata,
    )
    assert not np.shares_memory(source, result.audio)
    source[0] = 0.9
    assert result.audio[0] == pytest.approx(0.1)
    with pytest.raises(TypeError):
        result.metadata["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        result.metadata["voice_level"]["mode"] = "raw"  # type: ignore[index]
    assert result.metadata["items"] == ("one",)

    output = tmp_path / "sub" / "voice.wav"
    result.save_wav(output)
    with wave.open(str(output), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 24_000
        assert handle.getnframes() == 2


@pytest.mark.parametrize(
    ("audio", "sample_rate"),
    [([], 24_000), ([float("nan")], 24_000), ([0.1], 0), ([0.1], True), ([0.1], 24_000.0)],
)
def test_synthesis_result_rejects_invalid_waveforms_and_rates(
    audio: object,
    sample_rate: object,
) -> None:
    with pytest.raises(ValueError):
        SynthesisResult(
            audio=np.asarray(audio),
            sample_rate=sample_rate,  # type: ignore[arg-type]
            model_id="nano-v2",
            voice="default",
            speed=1.0,
            variation=0.667,
            seed=0,
        )
