from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ProviderUnavailableError

_PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
}


def _resolve_providers(ort: Any, providers: Sequence[str] | str | None) -> list[str]:
    available = list(ort.get_available_providers())
    if providers is None:
        requested = ["CPUExecutionProvider"]
    elif isinstance(providers, str):
        requested = [_PROVIDER_ALIASES.get(providers.lower(), providers)]
    else:
        requested = [_PROVIDER_ALIASES.get(value.lower(), value) for value in providers]
    for provider in requested:
        if provider not in available:
            raise ProviderUnavailableError(
                f"Provider {provider!r} is unavailable. Installed providers: {available}"
            )
    if requested[0] != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        requested.append("CPUExecutionProvider")
    return requested


class InflectOnnxRuntime:
    def __init__(
        self,
        duration_path: str | Path,
        decode_path: str | Path,
        *,
        providers: Sequence[str] | str | None = None,
        session_options: Any | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ProviderUnavailableError(
                "ONNX Runtime is not installed; install inflectsynth[cpu], [gpu], or [directml]"
            ) from exc
        selected = _resolve_providers(ort, providers)
        kwargs = {"providers": selected}
        if session_options is not None:
            kwargs["sess_options"] = session_options
        self.duration = ort.InferenceSession(str(duration_path), **kwargs)
        self.decode = ort.InferenceSession(str(decode_path), **kwargs)
        self.providers = tuple(selected)

    def infer_chunk(
        self,
        token_ids: tuple[int, ...],
        *,
        speed: float,
        variation: float,
        seed: int,
    ) -> np.ndarray:
        tokens = np.asarray(token_ids, dtype=np.int64)[None, :]
        m_p_exp, logs_p_exp, y_mask = self.duration.run(
            ["m_p_exp", "logs_p_exp", "y_mask"],
            {
                "tokens": tokens,
                "lengths": np.asarray([tokens.shape[1]], dtype=np.int64),
                "length_scale": np.asarray(1.0 / speed, dtype=np.float32),
            },
        )
        rng = np.random.default_rng(seed)
        latent_noise = rng.standard_normal(m_p_exp.shape, dtype=np.float32)
        waveform = self.decode.run(
            ["waveform"],
            {
                "m_p_exp": m_p_exp,
                "logs_p_exp": logs_p_exp,
                "y_mask": y_mask,
                "zp_noise": latent_noise,
                "noise_scale": np.asarray(variation, dtype=np.float32),
            },
        )[0]
        return np.asarray(waveform, dtype=np.float32).reshape(-1)
