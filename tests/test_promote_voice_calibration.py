from __future__ import annotations

import json

import pytest

from benchmarks import promote_voice_calibration as promoter
from benchmarks import voice_level_benchmark as benchmark
from inflectsynth import VoiceCalibrationKey, load_voice_calibration


def _report(model_id: str = "nano-v2", *, values: list[float] | None = None) -> dict[str, object]:
    entry = {
        "model_source": "inflect",
        "requested_model": model_id,
        "model_id": model_id,
        "model_ref": f"inflect:{model_id}",
        "voice": "default",
        "calibration_key": str(VoiceCalibrationKey("inflect", model_id, "default")),
    }
    stimuli = benchmark.load_stimuli()["stimuli"]
    loudness = values or [-22.0, -22.0, -22.0]
    rows = []
    for stimulus in stimuli:
        for seed, measured in enumerate(loudness):
            rows.append(
                {
                    **entry,
                    "stimulus_id": stimulus["id"],
                    "seed": seed,
                    "requested_speed": 1.0,
                    "variation": 0.667,
                    "sample_rate": 24_000,
                    "duration_seconds": 1.0,
                    "integrated_lufs": measured,
                    "raw_peak": 0.2,
                    "calibration_mode": "off",
                }
            )
    return benchmark.build_report(
        [entry], rows, [], stimuli=stimuli, policy=benchmark.load_policy()
    )


def test_candidate_contains_only_measured_complete_records_and_loads(tmp_path) -> None:
    report = _report()
    candidate, counts = promoter.build_candidate(report)
    key = "inflect:nano-v2:default"
    assert counts == {"eligible": 1, "high_variability": 0}
    assert candidate["voices"][key] == {
        "gain_db": -2.0,
        "measured_lufs": -22.0,
        "reference_lufs": -24.0,
        "mad_lu": 0.0,
        "samples": 9,
        "method": "bs1770",
        "corpus_version": "inflectsynth-prepared-speech-v1",
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    catalog = load_voice_calibration(path)
    assert catalog.voices[VoiceCalibrationKey.parse(key)].gain_db == -2.0


def test_promotion_rejects_incomplete_reports_and_measurement_failures() -> None:
    incomplete = _report()
    incomplete["coverage"]["complete"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="coverage is incomplete"):
        promoter.build_candidate(incomplete)

    failed = _report()
    failed["failures"] = [{"phase": "synthesis"}]
    with pytest.raises(ValueError, match="contains failures"):
        promoter.build_candidate(failed)


def test_high_variability_needs_explicit_review() -> None:
    report = _report(values=[-22.0, -20.0, -18.0])
    with pytest.raises(ValueError, match="high seed variability"):
        promoter.build_candidate(report)
    candidate, counts = promoter.build_candidate(report, include_high_variability=True)
    assert counts["high_variability"] == 1
    assert "inflect:nano-v2:default" in candidate["voices"]


def test_promotion_detects_modified_raw_measurements() -> None:
    report = _report()
    report["measurements"][0]["calibration_mode"] = "calibrated"  # type: ignore[index]
    with pytest.raises(ValueError, match="enabled calibration"):
        promoter.build_candidate(report)


def test_reports_merge_only_when_disjoint_and_matching_provenance() -> None:
    nano = _report("nano-v2")
    micro = _report("micro-v2")
    candidate, counts = promoter.build_candidate_from_reports([nano, micro])
    assert set(candidate["voices"]) == {
        "inflect:nano-v2:default",
        "inflect:micro-v2:default",
    }
    assert counts == {"eligible": 2, "high_variability": 0}
    with pytest.raises(ValueError, match="duplicate calibration key"):
        promoter.build_candidate_from_reports([nano, nano])


def test_promoter_never_writes_to_production_catalog_or_report(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    with pytest.raises(SystemExit, match="never writes directly"):
        promoter.main([str(report_path), "--output", str(promoter.PRODUCTION_CATALOG)])
    with pytest.raises(SystemExit, match="must not overwrite"):
        promoter.main([str(report_path), "--output", str(report_path), "--force"])
