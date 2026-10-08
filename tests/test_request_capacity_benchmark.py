from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from benchmarks import request_capacity_benchmark as benchmark
from inflectsynth import RequestMeasure


class FakeModel:
    model_id = "nano-v2"
    model_ref = "inflect:nano-v2"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.installed = SimpleNamespace()

    def measure_prepared(self, text: str) -> RequestMeasure:
        return RequestMeasure(
            fits=None,
            amount=len(text.split()),
            maximum=None,
            model_id=self.model_id,
        )

    def synthesize_prepared(self, text: str, *, voice: str, config: Any) -> Any:
        self.calls.append({"text": text, "voice": voice, "config": config})
        if self.fail:
            raise RuntimeError("synthetic failure")
        return SimpleNamespace(
            audio=np.asarray([0.1, -0.1, 0.05], dtype=np.float32),
            sample_rate=24_000,
            metadata={"normalized_text": text.strip(), "token_count": len(text.split())},
        )


def test_policy_and_stimuli_cover_required_models_shapes_targets_and_seeds() -> None:
    policy = benchmark.load_policy()
    stimuli = benchmark.load_stimuli()
    assert policy["models"] == ["nano-v2", "micro-v2"]
    assert policy["target_token_counts"] == [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400]
    assert policy["semantic_shapes"] == ["one_sentence", "multiple_sentences"]
    assert len(policy["seeds"]) >= 3
    assert len(stimuli["phrases"]) >= 4


def test_text_targets_are_chosen_from_frontend_token_measurements() -> None:
    model = FakeModel()
    text, amount = benchmark.choose_text_for_token_target(
        model, ["one two three", "four five"], target=10, shape="multiple_sentences"
    )
    assert amount == model.measure_prepared(text).amount
    assert text.count(".") >= 2
    assert abs(amount - 10) <= 3


def test_sample_record_contains_runtime_pcm_and_token_evidence() -> None:
    model = FakeModel()
    policy = benchmark.load_policy()
    text = "A few prepared words."

    row = benchmark.measure_sample(
        model,
        model_id="nano-v2",
        semantic_shape="one_sentence",
        target_token_count=10,
        text=text,
        seed=2,
        policy=policy,
    )

    assert row["synthesis_success"] is True
    assert row["token_count"] == len(text.split())
    assert row["source_text_length"] == len(text)
    assert row["normalized_text_length"] == len(text.strip())
    assert row["audio_samples"] == 3
    assert row["audio_seconds"] == pytest.approx(3 / 24_000)
    assert row["real_seconds"] >= 0
    assert row["xrt"] > 0
    assert row["sample_rate"] == 24_000
    assert row["finite_pcm"] is True
    assert row["peak_abs"] == pytest.approx(0.1)
    assert row["exception_type"] is None
    assert model.calls[0]["config"].voice_level.mode == "off"


def test_sample_failures_are_recorded_without_being_promoted() -> None:
    row = benchmark.measure_sample(
        FakeModel(fail=True),
        model_id="nano-v2",
        semantic_shape="one_sentence",
        target_token_count=10,
        text="A request.",
        seed=0,
        policy=benchmark.load_policy(),
    )
    assert row["synthesis_success"] is False
    assert row["exception_type"] == "RuntimeError"
    assert row["exception_phase"] == "synthesis"
    assert row["token_count"] == 2


def test_report_coverage_requires_the_complete_matrix() -> None:
    policy = benchmark.load_policy()
    report = benchmark.build_report(policy, benchmark.load_stimuli(), [], [], [])
    assert report["coverage"]["complete"] is False
    assert report["coverage"]["samples_expected"] == 132


def test_real_model_execution_is_explicitly_gated() -> None:
    with pytest.raises(SystemExit, match="pass --run-real-models explicitly"):
        benchmark.main([])
