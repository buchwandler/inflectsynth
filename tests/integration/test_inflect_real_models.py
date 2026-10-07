"""Release smoke tests for the real pinned Inflect v2 ONNX bundles.

These tests are deliberately opt-in. Install the ONNXVoice candidate and
``inflectsynth[cpu]`` first, then run from the InflectSynth checkout with:

    INFLECTSYNTH_RUN_REAL_MODEL_TESTS=1 python -m pytest -s \
        -m "integration and network" tests/integration/test_inflect_real_models.py

The first run downloads the pinned graphs from the configured catalog.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

_RUN_REAL_MODELS = os.environ.get("INFLECTSYNTH_RUN_REAL_MODEL_TESTS", "").casefold() in {
    "1",
    "true",
    "yes",
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.skipif(
        not _RUN_REAL_MODELS,
        reason="set INFLECTSYNTH_RUN_REAL_MODEL_TESTS=1 to download and run real model bundles",
    ),
]

_TEXT = "A small release smoke test for the Inflect ONNX runtime."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"expected installed model artifact at {path}"
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


@pytest.mark.parametrize("model_id", ("nano-v2", "micro-v2"))
def test_real_cpu_model_release_smoke(model_id: str) -> None:
    import numpy as np
    import onnxruntime
    import onnxvoice

    from inflectsynth import InflectVoice

    controls = {"speed": 1.0, "variation": 0.667}
    with InflectVoice.from_pretrained(
        model_id,
        providers="cpu",
        refresh_catalog=True,
    ) as tts:
        first = tts.synthesize(_TEXT, **controls, seed=7)
        repeated = tts.synthesize(_TEXT, **controls, seed=7)
        other_seed = tts.synthesize(_TEXT, **controls, seed=8)

        for result in (first, repeated, other_seed):
            assert result.sample_rate == 24_000
            assert result.audio.dtype == np.float32
            assert result.audio.ndim == 1
            assert result.audio.size > 0
            assert np.all(np.isfinite(result.audio))
        assert np.array_equal(first.audio, repeated.audio), (
            "same-seed CPU output was not repeatable"
        )
        assert not np.array_equal(first.audio, other_seed.audio), (
            "different seed did not change audio"
        )

        revision = first.metadata.get("revision")
        if not isinstance(revision, str):
            upstream = tts.metadata.get("upstream", {})
            revision = upstream.get("source_revision", upstream.get("revision"))
        assert isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision), (
            f"expected full pinned upstream revision, got {revision!r}"
        )

        artifacts = {
            "duration": _artifact_record(tts.installed.duration_path),
            "decode": _artifact_record(tts.installed.decode_path),
        }
        runtime_diagnostics = tts.diagnostics().get("runtime")
        report = {
            "model_id": tts.model_id,
            "model_ref": first.model_ref,
            "upstream_revision": revision,
            "artifacts": artifacts,
            "sample_rate": first.sample_rate,
            "provider_requested": "cpu",
            "onnxruntime_version": onnxruntime.__version__,
            "onnxvoice_version": onnxvoice.__version__,
            "inflectsynth_version": importlib.metadata.version("inflectsynth"),
            "runtime_diagnostics": runtime_diagnostics,
        }
        print(json.dumps(report, sort_keys=True, default=str))
