from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from inflectg2p import InflectG2P

from ._onnxvoice import (
    ResolvedInflectModel,
    install_pretrained_model,
    installation_to_model,
    open_installed_model,
    open_local_model,
)
from .config import SynthesisConfig
from .errors import (
    EmptyTextError,
    InflectSynthError,
    InvalidSynthesisConfigError,
    InvalidVoiceError,
    ModelInferenceError,
)
from .types import SynthesisResult, VoiceInfo
from .voice_level import (
    VoiceCalibrationKey,
    VoiceLevelApplication,
    apply_voice_level_calibration,
)

SAMPLE_RATE = 24_000


class InflectVoice:
    """Inflect v2 prepared-text synthesis engine using ONNXVoice for inference."""

    def __init__(
        self,
        *,
        runtime: Any,
        installed: ResolvedInflectModel,
        g2p: InflectG2P | None = None,
    ) -> None:
        self.runtime = runtime
        self.installed = installed
        self.model_ref = installed.ref
        self.model_id = installed.model_id
        self.metadata = dict(installed.metadata)
        self.sample_rate = installed.sample_rate
        self._owns_g2p = g2p is None
        self.g2p = g2p if g2p is not None else InflectG2P()
        self._closed = False
        self._last_voice_level_application: VoiceLevelApplication | None = None

    @classmethod
    def from_pretrained(
        cls,
        model: str = "nano-v2",
        *,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        refresh_catalog: bool = False,
        force_download: bool = False,
        catalog_url: str | None = None,
        providers: Sequence[Any] | str | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
        progress: Any | None = None,
        g2p: InflectG2P | None = None,
    ) -> InflectVoice:
        installed = install_pretrained_model(
            model,
            cache_dir=cache_dir,
            offline=offline,
            refresh_catalog=refresh_catalog,
            force_download=force_download,
            catalog_url=catalog_url,
            progress=progress,
        )
        runtime = open_installed_model(
            installed,
            cache_dir=cache_dir,
            offline=offline,
            catalog_url=catalog_url,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        return cls(runtime=runtime, installed=installed, g2p=g2p)

    @classmethod
    def from_local(
        cls,
        *,
        duration_path: str | Path,
        decode_path: str | Path,
        model_id: str = "local-inflect-v2",
        providers: Sequence[Any] | str | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, dict[str, Any]] | None = None,
        session_options: Any | None = None,
        g2p: InflectG2P | None = None,
    ) -> InflectVoice:
        metadata = {
            "id": model_id,
            "sample_rate": SAMPLE_RATE,
            "default_voice": "default",
            "voices": {
                "default": {
                    "id": "default",
                    "name": "Default",
                    "language": "en-US",
                    "locale": "en-US",
                    "gender": "unknown",
                    "synthetic": True,
                }
            },
            "runtime": {"profile": "inflect-v2-split-v1", "layout": "split"},
        }
        runtime = open_local_model(
            duration_path=duration_path,
            decode_path=decode_path,
            sample_rate=SAMPLE_RATE,
            metadata=metadata,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        installed = replace(
            installation_to_model(runtime.installation),
            ref=None,
            model_id=model_id,
        )
        return cls(runtime=runtime, installed=installed, g2p=g2p)

    @property
    def available_voices(self) -> tuple[str, ...]:
        voices = self.metadata.get("voices", {"default": {}})
        return tuple(voices) if isinstance(voices, Mapping) else ("default",)

    @property
    def voices(self) -> tuple[VoiceInfo, ...]:
        voices = self.metadata.get("voices", {})
        if not isinstance(voices, Mapping):
            return ()
        return tuple(
            VoiceInfo(
                id=str(voice_id),
                name=str(value.get("name", voice_id)),
                language=str(value.get("locale") or value.get("language", "en-US")),
                gender=str(value.get("gender", "unknown")),
                synthetic=bool(value.get("synthetic", True)),
            )
            for voice_id, value in voices.items()
            if isinstance(value, Mapping)
        )

    def calibration_key(self, voice: str) -> VoiceCalibrationKey | None:
        if self.model_ref is None:
            return None
        if voice not in self.available_voices:
            raise InvalidVoiceError(f"unknown Inflect voice: {voice!r}")
        return VoiceCalibrationKey(model_source="inflect", model_id=self.model_id, voice=voice)

    @property
    def last_voice_level_application(self) -> VoiceLevelApplication | None:
        return self._last_voice_level_application

    def synthesize_prepared(
        self,
        text: str,
        *,
        voice: str = "default",
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
        config: SynthesisConfig | None = None,
    ) -> SynthesisResult:
        if self._closed:
            raise RuntimeError("InflectVoice is closed")
        if not isinstance(text, str) or not text.strip():
            raise EmptyTextError("prepared text must be a non-empty string")
        if voice not in self.available_voices:
            raise InvalidVoiceError(
                "Inflect v2 base releases contain one fixed voice; voice must be 'default'"
            )
        if config is not None and not isinstance(config, SynthesisConfig):
            raise InvalidSynthesisConfigError("config must be a SynthesisConfig")
        synthesis_config = (
            config.validated()
            if config is not None
            else SynthesisConfig(speed=speed, variation=variation, seed=seed).validated()
        )

        try:
            frontend = self.g2p.phonemize_prepared(text)
        except InflectSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("InflectG2P failed to phonemize prepared text") from exc
        try:
            inference = self.runtime.infer(
                frontend.token_ids,
                speed=synthesis_config.speed,
                variation=synthesis_config.variation,
                seed=synthesis_config.seed,
            )
            raw_audio = inference.audio
        except InflectSynthError:
            raise
        except Exception as exc:
            raise ModelInferenceError("Inflect model inference failed") from exc

        try:
            audio = np.asarray(raw_audio, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelInferenceError("Inflect runtime returned invalid audio") from exc
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            raise ModelInferenceError("Inflect runtime audio must be non-empty and finite")

        audio, application = apply_voice_level_calibration(
            audio,
            synthesis_config.voice_level,
            self.calibration_key(voice),
        )
        self._last_voice_level_application = application
        upstream = self.metadata.get("upstream")
        upstream = upstream if isinstance(upstream, Mapping) else {}
        revision = (
            self.metadata.get("source_revision")
            or upstream.get("source_revision")
            or upstream.get("revision")
        )
        runtime_sample_rate = getattr(inference, "sample_rate", None)
        if runtime_sample_rate is None:
            runtime_sample_rate = self.sample_rate
        if (
            isinstance(runtime_sample_rate, bool)
            or not isinstance(runtime_sample_rate, Integral)
            or runtime_sample_rate <= 0
        ):
            raise ModelInferenceError("ONNXVoice returned an invalid sample rate")
        return SynthesisResult(
            audio=audio,
            sample_rate=int(runtime_sample_rate),
            model_id=self.model_id,
            voice=voice,
            speed=synthesis_config.speed,
            variation=synthesis_config.variation,
            seed=synthesis_config.seed,
            model_ref=self.model_ref,
            metadata={
                "normalized_text": frontend.text,
                "phoneme_text": frontend.phonemes,
                "token_count": len(frontend.token_ids),
                "model_id": self.model_id,
                "revision": revision,
                "voice_level": {
                    "mode": application.mode,
                    "applied": application.applied,
                    "gain_db": application.gain_db,
                    "source": application.source,
                    "calibration_key": str(application.key) if application.key else None,
                    "reason": application.reason,
                    "catalog_revision": application.catalog_revision,
                },
            },
        )

    synthesize_text = synthesize_prepared
    synthesize = synthesize_prepared

    def diagnostics(self) -> dict[str, object]:
        runtime_diagnostics = getattr(self.runtime, "diagnostics", None)
        g2p_diagnostics = getattr(self.g2p, "diagnostics", None)
        return {
            "runtime": (
                runtime_diagnostics() if callable(runtime_diagnostics) else runtime_diagnostics
            ),
            "g2p": g2p_diagnostics() if callable(g2p_diagnostics) else g2p_diagnostics,
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            close_runtime = getattr(self.runtime, "close", None)
            if callable(close_runtime):
                close_runtime()
        finally:
            try:
                if self._owns_g2p:
                    self.g2p.close()
            finally:
                self._closed = True

    def __enter__(self) -> InflectVoice:
        if self._closed:
            raise RuntimeError("InflectVoice is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
