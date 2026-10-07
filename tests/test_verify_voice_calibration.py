from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks import verify_voice_calibration as verifier
from inflectsynth import (
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelCalibration,
)


def _catalog() -> VoiceCalibrationCatalog:
    voices = {
        VoiceCalibrationKey("inflect", model_id, "default"): VoiceLevelCalibration(
            gain_db=-2.0,
            measured_lufs=-22.0,
            reference_lufs=-24.0,
            mad_lu=0.0,
            samples=9,
            corpus_version="inflectsynth-prepared-speech-v1",
        )
        for model_id in ("nano-v2", "micro-v2")
    }
    return VoiceCalibrationCatalog(
        schema=1,
        method="bs1770",
        corpus="inflectsynth-prepared-speech-v1",
        reference_lufs=-24.0,
        generated_with={"inflectsynth": "0.1.0"},
        voices=voices,
        revision="a" * 64,
    )


class FakeModel:
    def __init__(
        self,
        model_id: str,
        *,
        source: str = "catalog",
        peak: float = 0.5,
        close_error: bool = False,
    ) -> None:
        self.model_id = model_id
        self.available_voices = ("default",)
        self.source = source
        self.peak = peak
        self.close_error = close_error
        self.calls: list[dict[str, Any]] = []
        self.closed = 0

    def calibration_key(self, voice: str) -> VoiceCalibrationKey:
        return VoiceCalibrationKey("inflect", self.model_id, voice)

    def synthesize_prepared(self, text: str, *, voice: str, config: Any) -> Any:
        self.calls.append({"text": text, "voice": voice, "config": config})
        key = str(self.calibration_key(voice))
        return SimpleNamespace(
            audio=np.full(128, self.peak, dtype=np.float32),
            sample_rate=24_000,
            metadata={
                "voice_level": {
                    "mode": "calibrated",
                    "source": self.source,
                    "calibration_key": key,
                    "catalog_revision": "a" * 64,
                    "gain_db": -2.0,
                }
            },
        )

    def close(self) -> None:
        self.closed += 1
        if self.close_error:
            raise RuntimeError("close failed")


def _factory_for(**options: Any):
    opened: list[FakeModel] = []

    def factory(model_id: str, **_kwargs: Any) -> FakeModel:
        model = FakeModel(model_id, **options)
        opened.append(model)
        return model

    return factory, opened


def test_verification_policy_covers_both_models_default_voice_and_all_seeds() -> None:
    policy = verifier.load_verification_policy()
    assert policy["models"] == ["nano-v2", "micro-v2"]
    assert policy["voices"] == ["default"]
    assert policy["seeds"] == [0, 1, 2]
    assert policy["max_post_calibration_abs_error_lu"] == 0.5
    assert policy["max_post_calibration_peak_dbfs"] == -1.0


def test_verifier_passes_both_models_and_checks_catalog_metadata(monkeypatch) -> None:
    factory, opened = _factory_for()
    monkeypatch.setattr(
        verifier,
        "measure_loudness",
        lambda audio, *, sample_rate: SimpleNamespace(integrated_lufs=-24.0),
    )
    report = verifier.run_verification(open_model=factory, catalog=_catalog())
    assert report["status"] == "passed"
    assert report["coverage"] == {
        "identities_expected": 2,
        "identities_measured": 2,
        "identities_passed": 2,
        "identities_failed": 0,
        "measurement_failures": 0,
        "failure_count": 0,
        "complete": True,
    }
    assert [model.model_id for model in opened] == ["nano-v2", "micro-v2"]
    assert all(len(model.calls) == 9 for model in opened)
    assert all([call["config"].seed for call in model.calls] == [0, 1, 2] * 3 for model in opened)
    assert all(
        call["config"].voice_level.mode == "calibrated" for model in opened for call in model.calls
    )
    assert all(model.closed == 1 for model in opened)
    assert all(row["abs_error_lu"] == 0.0 for row in report["identities"])


def test_verifier_fails_loudness_peak_and_wrong_runtime_metadata(monkeypatch) -> None:
    factory, _ = _factory_for(source="override", peak=0.95)
    monkeypatch.setattr(
        verifier,
        "measure_loudness",
        lambda audio, *, sample_rate: SimpleNamespace(integrated_lufs=-23.0),
    )
    report = verifier.run_verification(open_model=factory, catalog=_catalog())
    assert report["status"] == "failed"
    assert report["coverage"]["complete"] is True
    assert report["coverage"]["identities_passed"] == 0
    assert all(row["status"] == "failed" for row in report["identities"])
    assert {failure["phase"] for failure in report["failures"]} == {"metadata", "acceptance"}


def test_verifier_marks_missing_catalog_identities_without_claiming_success() -> None:
    empty = VoiceCalibrationCatalog(
        schema=1,
        method="bs1770",
        corpus="inflectsynth-prepared-speech-v1",
        reference_lufs=-24.0,
        generated_with={},
        voices={},
        revision="b" * 64,
    )
    report = verifier.run_verification(catalog=empty)
    assert report["status"] == "failed"
    assert report["coverage"]["identities_measured"] == 0
    assert {failure["phase"] for failure in report["failures"]} == {"catalog"}


def test_model_open_and_close_failures_are_recorded(monkeypatch) -> None:
    def fail_open(_model_id: str, **_kwargs: Any) -> Any:
        raise RuntimeError("open failed")

    report = verifier.run_verification(open_model=fail_open, catalog=_catalog())
    assert report["status"] == "failed"
    assert report["coverage"]["identities_measured"] == 0
    assert len(report["failures"]) == 2
    assert all(row["phase"] == "model_open" for row in report["failures"])

    factory, _ = _factory_for(close_error=True)
    monkeypatch.setattr(
        verifier,
        "measure_loudness",
        lambda audio, *, sample_rate: SimpleNamespace(integrated_lufs=-24.0),
    )
    report = verifier.run_verification(open_model=factory, catalog=_catalog())
    assert report["status"] == "failed"
    assert all(row["status"] == "failed" for row in report["identities"])
    assert any(row["phase"] == "model_close" for row in report["failures"])


def test_verifier_cli_writes_report_and_returns_failure_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = {
        "status": "failed",
        "coverage": {"identities_passed": 0, "identities_expected": 2, "measurement_failures": 1},
    }
    monkeypatch.setattr(verifier, "run_verification", lambda **_kwargs: report)
    output = tmp_path / "verification.json"
    assert verifier.main(["--output", str(output)]) == 1
    assert output.is_file()
