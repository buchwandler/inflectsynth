from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from inflectg2p import InflectG2P

from .errors import InvalidSynthesisConfigError, InvalidVoiceError
from .install import install_model
from .runtime import InflectOnnxRuntime
from .types import InstalledModel, SynthesisResult, VoiceInfo

SAMPLE_RATE = 24_000


def split_text(text: str, limit: int = 280) -> list[str]:
    normalized = " ".join(text.split())
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;:])\s+", normalized) if part.strip()]
    chunks: list[str] = []
    for sentence in sentences or [normalized]:
        while len(sentence) > limit:
            search = sentence[: limit + 1]
            punctuation = max(search.rfind(mark) for mark in (",", ";", ":"))
            split_at = (
                punctuation + 1 if punctuation >= limit // 2 else sentence.rfind(" ", 0, limit + 1)
            )
            if split_at < limit // 2:
                split_at = limit
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def boundary_pause_seconds(chunk: str) -> float:
    ending = chunk.rstrip()[-1:] if chunk.strip() else ""
    return {"?": 0.28, "!": 0.24, ".": 0.22, ";": 0.16, ":": 0.13, ",": 0.09}.get(ending, 0.08)


def edge_fade(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    milliseconds: float = 5.0,
) -> np.ndarray:
    frames = min(round(sample_rate * milliseconds / 1000.0), waveform.size // 2)
    if frames <= 0:
        return waveform
    output = waveform.copy()
    ramp = np.linspace(0.0, 1.0, frames, endpoint=True, dtype=np.float32)
    output[:frames] *= ramp
    output[-frames:] *= ramp[::-1]
    return output


class InflectVoice:
    """Inflect v2 synthesis engine for Nano v2 and Micro v2.

    Both upstream base releases have one fixed synthetic English voice. The public
    ``voice`` parameter therefore accepts only ``"default"``.
    """

    def __init__(
        self,
        *,
        runtime: InflectOnnxRuntime,
        installed: InstalledModel,
        g2p: InflectG2P | None = None,
    ) -> None:
        self.runtime = runtime
        self.installed = installed
        self.model_id = installed.model_id
        self.metadata = installed.metadata
        self.g2p = g2p or InflectG2P()

    @classmethod
    def from_pretrained(
        cls,
        model: str = "nano-v2",
        *,
        cache_dir: str | Path | None = None,
        catalog_path: str | Path | None = None,
        offline: bool = False,
        force_download: bool = False,
        providers: Sequence[str] | str | None = None,
        session_options: Any | None = None,
        g2p: InflectG2P | None = None,
    ) -> InflectVoice:
        installed = install_model(
            model,
            cache_dir=cache_dir,
            catalog_path=catalog_path,
            offline=offline,
            force_download=force_download,
        )
        runtime = InflectOnnxRuntime(
            installed.duration_path,
            installed.decode_path,
            providers=providers,
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
        providers: Sequence[str] | str | None = None,
        session_options: Any | None = None,
        g2p: InflectG2P | None = None,
    ) -> InflectVoice:
        root = Path(duration_path).resolve().parent
        installed = InstalledModel(
            model_id=model_id,
            root=root,
            duration_path=Path(duration_path),
            decode_path=Path(decode_path),
            metadata={
                "id": model_id,
                "sample_rate": SAMPLE_RATE,
                "default_voice": "default",
                "voices": {
                    "default": {
                        "id": "default",
                        "name": "Default",
                        "language": "en-US",
                        "gender": "unknown",
                        "synthetic": True,
                    }
                },
            },
        )
        runtime = InflectOnnxRuntime(
            duration_path,
            decode_path,
            providers=providers,
            session_options=session_options,
        )
        return cls(runtime=runtime, installed=installed, g2p=g2p)

    @property
    def available_voices(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("voices", {"default": {}}))

    @property
    def voices(self) -> tuple[VoiceInfo, ...]:
        output: list[VoiceInfo] = []
        for voice_id, value in self.metadata.get("voices", {}).items():
            output.append(
                VoiceInfo(
                    id=voice_id,
                    name=value.get("name", voice_id),
                    language=value.get("language", "en-US"),
                    gender=value.get("gender", "unknown"),
                    synthetic=bool(value.get("synthetic", True)),
                )
            )
        return tuple(output)

    def synthesize(
        self,
        text: str,
        *,
        voice: str = "default",
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
    ) -> SynthesisResult:
        normalized_input = " ".join(text.split()) if isinstance(text, str) else ""
        if not normalized_input:
            raise InvalidSynthesisConfigError("text must not be empty")
        if voice != "default":
            raise InvalidVoiceError(
                "Inflect v2 base releases contain one fixed voice; voice must be 'default'"
            )
        speed = float(speed)
        variation = float(variation)
        if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
            raise InvalidSynthesisConfigError("speed must be finite and between 0.5 and 2.0")
        if not math.isfinite(variation) or not 0.0 <= variation <= 1.0:
            raise InvalidSynthesisConfigError("variation must be finite and between 0.0 and 1.0")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise InvalidSynthesisConfigError("seed must be an integer")

        chunks = split_text(normalized_input)
        pieces: list[np.ndarray] = []
        chunk_metadata: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            if index:
                pause = boundary_pause_seconds(chunks[index - 1])
                pieces.append(np.zeros(round(SAMPLE_RATE * pause), dtype=np.float32))
            frontend = self.g2p.phonemize(chunk)
            waveform = self.runtime.infer_chunk(
                frontend.token_ids,
                speed=speed,
                variation=variation,
                seed=seed + index,
            )
            pieces.append(edge_fade(waveform, SAMPLE_RATE))
            chunk_metadata.append(
                {
                    "text": chunk,
                    "normalized_text": frontend.normalized_text,
                    "phoneme_text": frontend.phoneme_text,
                    "token_count": len(frontend.token_ids),
                    "seed": seed + index,
                }
            )
        audio = np.clip(np.concatenate(pieces), -1.0, 1.0)
        return SynthesisResult(
            audio=audio,
            sample_rate=int(self.metadata.get("sample_rate", SAMPLE_RATE)),
            model_id=self.model_id,
            voice=voice,
            speed=speed,
            variation=variation,
            seed=seed,
            metadata={
                "chunks": chunk_metadata,
                "providers": list(self.runtime.providers),
                "revision": (self.metadata.get("upstream") or {}).get("revision"),
            },
        )

    # Compatibility aliases for callers familiar with sibling synth packages.
    synthesize_text = synthesize
    synthesize_prepared = synthesize

    def close(self) -> None:
        return None

    def __enter__(self) -> InflectVoice:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
