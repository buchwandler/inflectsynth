from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks import voice_level_benchmark as benchmark
from inflectsynth import VoiceCalibrationKey


class FakeModel:
    model_id = "nano-v2"
    model_ref = "inflect:nano-v2"
    available_voices = ("default",)

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def synthesize_prepared(self, text: str, *, voice: str, config: Any) -> Any:
        self.calls.append({"text": text, "voice": voice, "config": config})
        return SimpleNamespace(
            audio=np.asarray([0.2, -0.2, 0.1], dtype=np.float32),
            sample_rate=24_000,
            duration_seconds=0.1,
        )


def _measurements(entry: dict[str, Any], centers: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for stimulus in benchmark.load_stimuli()["stimuli"]:
        for seed, loudness in enumerate(centers[stimulus["id"]]):
            rows.append(
                {
                    **entry,
                    "stimulus_id": stimulus["id"],
                    "seed": seed,
                    "requested_speed": 1.0,
                    "variation": 0.667,
                    "sample_rate": 24_000,
                    "duration_seconds": 1.0,
                    "integrated_lufs": loudness,
                    "raw_peak": 0.2,
                    "calibration_mode": "off",
                }
            )
    return rows


def test_policy_and_stimuli_declare_required_seeded_corpus() -> None:
    policy = benchmark.load_policy()
    document = benchmark.load_stimuli()
    assert policy["seeds"] == [0, 1, 2]
    assert [item["id"] for item in document["stimuli"]] == ["short", "medium", "long"]
    assert all(item["text"].strip() for item in document["stimuli"])


def test_policy_rejects_wrong_seed_coverage(tmp_path) -> None:
    policy = benchmark.load_policy()
    policy["seeds"] = [0, 0, 2]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match=r"exactly \[0, 1, 2\]"):
        benchmark.load_policy(path)


def test_identity_expansion_uses_only_stable_default_model_voice() -> None:
    model = FakeModel()
    entry = benchmark.expand_identities(model, requested_model="nano-v2")[0]
    assert entry["calibration_key"] == str(VoiceCalibrationKey("inflect", "nano-v2", "default"))
    assert entry["model_ref"] == "inflect:nano-v2"
    with pytest.raises(ValueError, match="unexpected ID"):
        benchmark.expand_identities(model, requested_model="micro-v2")


def test_measurement_uses_each_seed_once_and_disables_runtime_calibration(monkeypatch) -> None:
    model = FakeModel()
    entry = benchmark.expand_identities(model, requested_model="nano-v2")
    monkeypatch.setattr(
        benchmark,
        "measure_loudness",
        lambda audio, *, sample_rate: SimpleNamespace(integrated_lufs=-22.0),
    )
    measurements, failures = benchmark.measure_identities(
        model,
        entry,
        benchmark.load_stimuli()["stimuli"],
        benchmark.load_policy(),
    )
    assert failures == []
    assert len(measurements) == 9
    assert [call["config"].seed for call in model.calls] == [0, 1, 2] * 3
    assert all(call["config"].speed == 1.0 for call in model.calls)
    assert all(call["config"].variation == 0.667 for call in model.calls)
    assert all(call["config"].voice_level.mode == "off" for call in model.calls)
    assert all(row["calibration_mode"] == "off" for row in measurements)


def test_seed_mads_and_clamped_gain_are_aggregated_per_identity() -> None:
    entry = benchmark.expand_identities(FakeModel(), requested_model="nano-v2")[0]
    policy = benchmark.load_policy()
    stimuli = benchmark.load_stimuli()["stimuli"]
    measurements = _measurements(
        entry,
        {
            "short": [-20.0, -21.0, -22.0],
            "medium": [-22.0, -22.1, -21.9],
            "long": [-24.0, -24.1, -23.9],
        },
    )
    aggregate = benchmark.aggregate_measurements([entry], measurements, stimuli, policy)[0]
    assert aggregate["complete"] is True
    assert aggregate["sample_count"] == 9
    assert aggregate["median_lufs"] == -22.0
    assert aggregate["mad_lu"] == 1.0
    assert aggregate["status"] == "high_variability"
    assert aggregate["gain_db"] == -2.0
    assert aggregate["predicted_peak"] == pytest.approx(0.2 * (10.0 ** (-2.0 / 20.0)))


def test_missing_seed_yields_incomplete_report_and_failure_coverage() -> None:
    entry = benchmark.expand_identities(FakeModel(), requested_model="nano-v2")[0]
    stimuli = benchmark.load_stimuli()["stimuli"]
    measurements = _measurements(
        entry,
        {stimulus["id"]: [-22.0, -22.0, -22.0] for stimulus in stimuli},
    )
    measurements.pop()
    report = benchmark.build_report(
        [entry],
        measurements,
        [],
        stimuli=stimuli,
        policy=benchmark.load_policy(),
    )
    assert report["coverage"]["complete"] is False
    assert report["coverage"]["identities_measured"] == 0
    assert report["aggregates"][0]["status"] == "incomplete"


def test_synthesis_failure_records_seed_and_phase(monkeypatch) -> None:
    class FailingModel(FakeModel):
        def synthesize_prepared(self, text: str, *, voice: str, config: Any) -> Any:
            raise RuntimeError("synthetic inference failure")

    model = FailingModel()
    entries = benchmark.expand_identities(model, requested_model="nano-v2")
    measurements, failures = benchmark.measure_identities(
        model,
        entries,
        benchmark.load_stimuli()["stimuli"][:1],
        benchmark.load_policy(),
    )
    assert measurements == []
    assert [row["seed"] for row in failures] == [0, 1, 2]
    assert all(row["phase"] == "synthesis" for row in failures)
