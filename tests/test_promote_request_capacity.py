from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from benchmarks import promote_request_capacity as promoter
from benchmarks import request_capacity_benchmark as benchmark


def _complete_report() -> dict[str, Any]:
    policy = benchmark.load_policy()
    stimuli = benchmark.load_stimuli()
    models = [
        {
            "model_id": model_id,
            "model_ref": f"inflect:{model_id}",
            "source_revision": f"revision-{model_id}",
            "artifacts": {
                name: {"bytes": 100, "sha256": "a" * 64} for name in ("duration", "decode")
            },
        }
        for model_id in policy["models"]
    ]
    samples = []
    for model_id in policy["models"]:
        for shape in policy["semantic_shapes"]:
            for target in policy["target_token_counts"]:
                for seed in policy["seeds"]:
                    samples.append(
                        {
                            "model": model_id,
                            "semantic_shape": shape,
                            "target_token_count": target,
                            "source_text_length": target * 2,
                            "normalized_text_length": target * 2,
                            "token_count": target,
                            "seed": seed,
                            "speed": policy["speed"],
                            "variation": policy["variation"],
                            "synthesis_success": True,
                            "exception_type": None,
                            "audio_samples": 24_000,
                            "audio_seconds": 1.0,
                            "real_seconds": 0.5,
                            "xrt": 0.5,
                            "sample_rate": policy["sample_rate"],
                            "finite_pcm": True,
                            "peak_abs": 0.5,
                        }
                    )
    report = benchmark.build_report(policy, stimuli, models, samples, [])
    report["generated_with"] = {
        "inflectsynth": "0.1.1",
        "inflectg2p": "0.1.0",
        "onnxvoice": "0.2.5",
    }
    report["runtime_identity"] = {
        "engine": "inflect",
        "engine_version": "0.1.1",
        "g2p_revision": "0.1.0",
        "runtime_revision": "0.2.5",
        "request_api_version": "1",
    }
    report["environment"] = {"python_version": "3.12.0", "platform": "release-test"}
    return report


def test_candidate_uses_reproducible_policy_and_per_model_safe_boundaries() -> None:
    report = _complete_report()
    candidate = promoter.build_candidate(report, {"nano-v2": 300, "micro-v2": 200})

    assert candidate["unit"] == "model_tokens"
    assert candidate["models"]["nano-v2"]["maximum"] == 300
    assert candidate["models"]["micro-v2"]["maximum"] == 200
    assert candidate["models"]["nano-v2"]["basis"] == "validated-safe-request-budget"
    assert candidate["models"]["nano-v2"]["safe_observation_boundary"] == 400
    assert len(candidate["revision"]) == 64
    assert len(candidate["evidence_sha256"]) == 64


def test_candidate_rejects_incomplete_failed_or_noncanonical_evidence() -> None:
    incomplete = _complete_report()
    incomplete["coverage"]["complete"] = False
    with pytest.raises(ValueError, match="coverage is incomplete"):
        promoter.build_candidate(incomplete, {"nano-v2": 100, "micro-v2": 100})

    failed = _complete_report()
    failed["samples"][0]["synthesis_success"] = False
    with pytest.raises(ValueError, match="failed synthesis"):
        promoter.build_candidate(failed, {"nano-v2": 100, "micro-v2": 100})

    noncanonical = _complete_report()
    noncanonical["models"][0]["model_id"] = "nano"
    with pytest.raises(ValueError, match="canonical"):
        promoter.build_candidate(noncanonical, {"nano-v2": 100, "micro-v2": 100})


def test_unsupported_android_runtime_cannot_promote_capacity() -> None:
    report = _complete_report()
    report["environment"]["platform"] = "Android-16-aarch64-64bit-ELF"
    with pytest.raises(ValueError, match="unsupported Android runtime"):
        promoter.build_candidate(report, {"nano-v2": 100, "micro-v2": 100})


def test_candidate_maximum_must_be_positive_and_within_each_safe_boundary() -> None:
    report = _complete_report()
    with pytest.raises(ValueError, match="positive integer"):
        promoter.build_candidate(report, {"nano-v2": 0, "micro-v2": 100})
    with pytest.raises(ValueError, match="safe observation boundary"):
        promoter.build_candidate(report, {"nano-v2": 401, "micro-v2": 100})


def test_report_validator_rejects_duplicate_and_missing_samples() -> None:
    duplicate = _complete_report()
    duplicate["samples"][-1] = deepcopy(duplicate["samples"][0])
    with pytest.raises(ValueError, match="duplicate"):
        promoter.build_candidate(duplicate, {"nano-v2": 100, "micro-v2": 100})

    missing = _complete_report()
    missing["samples"].pop()
    with pytest.raises(ValueError, match="sample count"):
        promoter.build_candidate(missing, {"nano-v2": 100, "micro-v2": 100})


def test_promotion_never_writes_production_policy(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="never overwrites production policy"):
        promoter.main(
            [
                str(report_path),
                "--maximum",
                "nano-v2=100",
                "--maximum",
                "micro-v2=100",
                "--output",
                str(promoter.PRODUCTION_POLICY),
            ]
        )
