from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from inflectsynth import InflectVoice, SynthesisConfig, VoiceLevelConfig
from inflectsynth._onnxvoice import ResolvedInflectModel
from inflectsynth.errors import (
    EmptyTextError,
    InflectSynthError,
    InvalidSeedError,
    InvalidSpeedError,
    InvalidSynthesisConfigError,
    InvalidVariationError,
    InvalidVoiceError,
    ModelInferenceError,
)


class FakeG2P:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = 0
        self.diagnostics = {"backend": "fake"}

    def phonemize_prepared(self, text: str) -> Any:
        self.calls.append(text)
        return SimpleNamespace(text=text, phonemes="həlˈoʊ", token_ids=(0, 1, 0))

    def close(self) -> None:
        self.closed += 1


class FakeRuntime:
    def __init__(
        self,
        audio: Any = (0.5, -0.25),
        error: Exception | None = None,
        sample_rate: Any = 24_000,
    ) -> None:
        self.audio = audio
        self.error = error
        self.sample_rate = sample_rate
        self.calls: list[tuple[tuple[int, ...], dict[str, Any]]] = []
        self.closed = 0
        self.diagnostics = {"components": ("duration", "decode")}

    def infer(self, token_ids: tuple[int, ...], **kwargs: Any) -> Any:
        self.calls.append((token_ids, kwargs))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            audio=np.asarray(self.audio, dtype=np.float32),
            sample_rate=self.sample_rate,
        )

    def close(self) -> None:
        self.closed += 1


def _model(*, managed: bool = True) -> ResolvedInflectModel:
    return ResolvedInflectModel(
        ref="inflect:nano-v2" if managed else None,
        model_id="nano-v2" if managed else "local-model",
        duration_path=Path("duration.onnx"),
        decode_path=Path("decode.onnx"),
        sample_rate=24_000,
        metadata={
            "id": "nano-v2",
            "source_revision": "source-rev",
            "voices": {
                "default": {
                    "id": "default",
                    "name": "Default",
                    "language": "en-US",
                    "locale": "en-US",
                    "gender": "male",
                    "synthetic": True,
                }
            },
        },
        installation=object(),
    )


def _voice(
    *, managed: bool = True, runtime: FakeRuntime | None = None, g2p: FakeG2P | None = None
) -> tuple[InflectVoice, FakeRuntime, FakeG2P]:
    selected_runtime = runtime or FakeRuntime()
    selected_g2p = g2p or FakeG2P()
    voice = InflectVoice(
        runtime=selected_runtime,
        installed=_model(managed=managed),
        g2p=selected_g2p,
    )
    return voice, selected_runtime, selected_g2p


def test_atomic_prepared_request_preserves_text_controls_and_applies_gain() -> None:
    long_text = "A caller-shaped prepared request. " * 24
    voice, runtime, g2p = _voice(runtime=FakeRuntime(audio=(0.8, -0.5)))
    config = SynthesisConfig(
        speed=1.25,
        variation=0.4,
        seed=37,
        voice_level=VoiceLevelConfig(gain_db=6.0),
    )

    result = voice.synthesize_prepared(
        long_text,
        speed=0.5,
        variation=0.1,
        seed=1,
        config=config,
    )

    assert g2p.calls == [long_text]
    assert len(runtime.calls) == 1
    token_ids, controls = runtime.calls[0]
    assert token_ids == (0, 1, 0)
    assert controls == {"speed": 1.25, "variation": 0.4, "seed": 37}
    assert result.audio.max() > 1.0  # static gain is not clipped in memory
    assert result.speed == 1.25
    assert result.variation == 0.4
    assert result.seed == 37
    assert result.metadata["normalized_text"] == long_text
    assert result.metadata["phoneme_text"] == "həlˈoʊ"
    assert result.metadata["voice_level"]["source"] == "override"
    assert voice.last_voice_level_application is not None
    assert voice.last_voice_level_application.applied is True


def test_config_is_authoritative_and_result_metadata_is_immutable() -> None:
    voice, runtime, _ = _voice()
    result = voice.synthesize_prepared(
        "Text.",
        speed=1.9,
        variation=0.9,
        seed=9,
        config=SynthesisConfig(speed=0.75, variation=0.25, seed=4),
    )
    assert runtime.calls[0][1] == {"speed": 0.75, "variation": 0.25, "seed": 4}
    with pytest.raises(TypeError):
        result.metadata["changed"] = True  # type: ignore[index]


def test_prepared_text_is_not_whitespace_collapsed() -> None:
    text = "  Keep   caller   whitespace.  "
    voice, _, g2p = _voice()
    voice.synthesize_prepared(text)
    assert g2p.calls == [text]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"text": ""}, EmptyTextError),
        ({"text": 3}, EmptyTextError),
        ({"text": "hello", "voice": "other"}, InvalidVoiceError),
        ({"text": "hello", "speed": True}, InvalidSpeedError),
        ({"text": "hello", "speed": float("nan")}, InvalidSpeedError),
        ({"text": "hello", "speed": 2.1}, InvalidSpeedError),
        ({"text": "hello", "variation": float("inf")}, InvalidVariationError),
        ({"text": "hello", "variation": -0.1}, InvalidVariationError),
        ({"text": "hello", "seed": True}, InvalidSeedError),
        ({"text": "hello", "seed": -1}, InvalidSeedError),
        ({"text": "hello", "config": object()}, InvalidSynthesisConfigError),
    ],
)
def test_validation_raises_typed_errors(kwargs: dict[str, Any], error: type[Exception]) -> None:
    voice, runtime, g2p = _voice()
    text = kwargs.pop("text")
    with pytest.raises(error):
        voice.synthesize_prepared(text, **kwargs)
    assert runtime.calls == []
    assert g2p.calls == []


def test_unknown_runtime_errors_are_translated_and_known_errors_pass_through() -> None:
    runtime = FakeRuntime(error=RuntimeError("provider detail"))
    voice, _, _ = _voice(runtime=runtime)
    with pytest.raises(ModelInferenceError) as caught:
        voice.synthesize_prepared("Hello.")
    assert isinstance(caught.value.__cause__, RuntimeError)

    known = InvalidVoiceError("known")
    runtime = FakeRuntime(error=known)
    voice, _, _ = _voice(runtime=runtime)
    with pytest.raises(InvalidVoiceError) as caught:
        voice.synthesize_prepared("Hello.")
    assert caught.value is known
    assert isinstance(known, InflectSynthError)


def test_calibration_identity_is_managed_only() -> None:
    managed, _, _ = _voice()
    local, _, _ = _voice(managed=False)
    assert str(managed.calibration_key("default")) == "inflect:nano-v2:default"
    assert local.calibration_key("default") is None


def test_diagnostics_and_idempotent_ownership_lifecycle() -> None:
    voice, runtime, g2p = _voice()
    diagnostics = voice.diagnostics()
    assert diagnostics["runtime"] == {"components": ("duration", "decode")}
    assert diagnostics["g2p"] == {"backend": "fake"}
    voice.close()
    voice.close()
    assert runtime.closed == 1
    assert g2p.closed == 0  # externally supplied G2P remains caller-owned
    with pytest.raises(RuntimeError, match="closed"):
        voice.synthesize_prepared("Hello.")


def test_in_memory_result_audio_is_owned_finite_and_mono(tmp_path: Path) -> None:
    source = np.asarray([1.5, -1.5], dtype=np.float32)
    voice, _, _ = _voice(runtime=FakeRuntime(audio=source))
    result = voice.synthesize_prepared("Hello.")
    assert result.audio.dtype == np.float32
    assert result.audio.ndim == 1
    assert np.isfinite(result.audio).all()
    assert not np.shares_memory(result.audio, source)
    assert result.duration_seconds == pytest.approx(2 / 24_000)
    output = tmp_path / "nested" / "speech.wav"
    result.save_wav(output)
    assert output.is_file()


def test_g2p_failures_are_translated_before_runtime_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    voice, runtime, g2p = _voice()

    def fail(_text: str) -> Any:
        raise RuntimeError("phonemizer detail")

    monkeypatch.setattr(g2p, "phonemize_prepared", fail)
    with pytest.raises(ModelInferenceError) as caught:
        voice.synthesize_prepared("Hello.")
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert runtime.calls == []


def test_engine_closes_only_the_g2p_instance_it_created(monkeypatch: pytest.MonkeyPatch) -> None:
    owned_g2p = FakeG2P()
    monkeypatch.setattr("inflectsynth.voice.InflectG2P", lambda: owned_g2p)
    voice = InflectVoice(runtime=FakeRuntime(), installed=_model())
    voice.close()
    voice.close()
    assert owned_g2p.closed == 1


@pytest.mark.parametrize("sample_rate", [0, True, 24_000.0])
def test_invalid_runtime_sample_rate_is_a_typed_inference_error(sample_rate: Any) -> None:
    voice, _, _ = _voice(runtime=FakeRuntime(sample_rate=sample_rate))
    with pytest.raises(ModelInferenceError, match="sample rate"):
        voice.synthesize_prepared("Hello.")
