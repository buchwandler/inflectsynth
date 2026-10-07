#!/usr/bin/env python3
"""Build a reviewable runtime catalog from complete real Inflect measurements."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from inflectsynth import VoiceCalibrationKey

from . import voice_level_benchmark as benchmark

PRODUCTION_CATALOG = (
    Path(__file__).resolve().parents[1] / "inflectsynth" / "data" / "voice_level_calibration.json"
)
DEFAULT_INPUT = Path("benchmarks/output/voice_level_calibration/measurements.json")
DEFAULT_OUTPUT = Path("benchmarks/output/voice_level_calibration/candidate_catalog.json")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_report(
    report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if report.get("schema") != 1 or isinstance(report.get("schema"), bool):
        raise ValueError("unsupported measurement report schema")
    policy = report.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("measurement report is missing its policy")
    canonical_policy = benchmark.load_policy()
    if dict(policy) != canonical_policy:
        raise ValueError("measurement report policy differs from the checked-in benchmark policy")
    if report.get("corpus") != policy.get("name"):
        raise ValueError("measurement report corpus does not match its policy")
    stimuli = report.get("stimuli")
    if stimuli != benchmark.load_stimuli()["stimuli"]:
        raise ValueError("measurement report stimuli differ from the checked-in prepared corpus")
    generated_with = report.get("generated_with")
    required_versions = {"inflectsynth", "inflectg2p", "onnxvoice", "audiosig"}
    if (
        not isinstance(generated_with, Mapping)
        or set(generated_with) != required_versions
        or any(not isinstance(value, str) or not value for value in generated_with.values())
    ):
        raise ValueError("measurement report generated_with versions are incomplete")
    coverage = report.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("complete") is not True:
        raise ValueError("measurement report coverage is incomplete")
    failures = report.get("failures")
    entries = report.get("aggregates")
    measurements = report.get("measurements")
    if not isinstance(failures, list) or failures:
        raise ValueError("measurement report contains failures")
    if not isinstance(entries, list) or any(not isinstance(item, Mapping) for item in entries):
        raise ValueError("measurement report aggregates must be a list of objects")
    if not isinstance(measurements, list) or any(
        not isinstance(item, Mapping) for item in measurements
    ):
        raise ValueError("measurement report measurements must be a list of objects")
    identities_expected = len(entries)
    expected_samples = identities_expected * len(stimuli) * len(policy["seeds"])
    if (
        coverage.get("identities_expected") != identities_expected
        or coverage.get("identities_measured") != identities_expected
        or coverage.get("identities_failed") != 0
        or coverage.get("measurements_expected") != expected_samples
        or coverage.get("measurements_measured") != expected_samples
        or coverage.get("measurement_failures") != 0
        or len(measurements) != expected_samples
    ):
        raise ValueError("measurement report coverage counts do not match all required samples")

    seen_keys: set[str] = set()
    for entry in entries:
        source = entry.get("model_source")
        model_id = entry.get("model_id")
        voice = entry.get("voice")
        if not all(isinstance(value, str) and value for value in (source, model_id, voice)):
            raise ValueError("aggregate identity fields must be non-empty strings")
        key = VoiceCalibrationKey(source, model_id, voice)
        if entry.get("calibration_key") != str(key) or str(key) in seen_keys:
            raise ValueError(f"invalid or duplicate calibration identity: {key}")
        if voice != "default":
            raise ValueError(f"unexpected calibration voice: {voice!r}")
        seen_keys.add(str(key))

    seen_samples: set[tuple[str, str, int]] = set()
    for row in measurements:
        key_text = row.get("calibration_key")
        if key_text not in seen_keys:
            raise ValueError(f"measurement references unknown calibration identity: {key_text!r}")
        stimulus_id = row.get("stimulus_id")
        seed = row.get("seed")
        if stimulus_id not in {item["id"] for item in stimuli}:
            raise ValueError(f"measurement references unknown stimulus: {stimulus_id!r}")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in policy["seeds"]:
            raise ValueError("measurement seed is outside the declared seed policy")
        sample_id = (str(key_text), str(stimulus_id), seed)
        if sample_id in seen_samples:
            raise ValueError(f"duplicate measurement sample: {sample_id}")
        seen_samples.add(sample_id)
        if row.get("calibration_mode") != "off":
            raise ValueError(f"measurement for {key_text} enabled calibration")
        if row.get("requested_speed") != 1.0 or row.get("variation") != 0.667:
            raise ValueError(f"measurement for {key_text} used unexpected synthesis controls")
        if isinstance(row.get("sample_rate"), bool) or not isinstance(row.get("sample_rate"), int):
            raise ValueError("measurement sample_rate must be a positive integer")
        if row["sample_rate"] <= 0:
            raise ValueError("measurement sample_rate must be positive")
        if _finite(row.get("duration_seconds"), "duration_seconds") <= 0:
            raise ValueError("measurement duration_seconds must be positive")
        _finite(row.get("integrated_lufs"), "integrated_lufs")
        if _finite(row.get("raw_peak"), "raw_peak") < 0:
            raise ValueError("measurement raw_peak must be non-negative")
    if len(seen_samples) != expected_samples:
        raise ValueError(
            "measurement report does not cover each identity/stimulus/seed combination"
        )

    recomputed = benchmark.aggregate_measurements(
        entries,
        measurements,
        stimuli,
        policy,
    )
    for declared, actual in zip(entries, recomputed, strict=True):
        for field in (
            "complete",
            "status",
            "sample_count",
            "stimulus_count",
            "gain_limited",
            "peak_safety_review_required",
        ):
            if declared.get(field) != actual.get(field):
                raise ValueError(
                    f"aggregate {declared['calibration_key']} field {field} is inconsistent"
                )
        for field in ("median_lufs", "mad_lu", "gain_db", "predicted_peak"):
            if not math.isclose(
                _finite(declared.get(field), field),
                _finite(actual.get(field), field),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"aggregate {declared['calibration_key']} field {field} is inconsistent"
                )
    return policy, entries, measurements


def build_candidate(
    report: Mapping[str, Any],
    *,
    include_high_variability: bool = False,
    allow_peak_risk: bool = False,
    allow_gain_limited: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Validate all real samples and derive measured runtime catalog records."""
    policy, aggregates, _ = _validate_report(report)
    generated_with = report["generated_with"]
    voices: dict[str, dict[str, Any]] = {}
    counts = {"eligible": 0, "high_variability": 0}
    for aggregate in aggregates:
        key = str(aggregate["calibration_key"])
        if aggregate.get("complete") is not True or aggregate.get("status") == "incomplete":
            raise ValueError(f"{key} has incomplete measurements")
        status = aggregate.get("status")
        if status not in {"eligible", "high_variability"}:
            raise ValueError(f"{key} has unknown measurement status: {status!r}")
        if status == "high_variability" and not include_high_variability:
            raise ValueError(f"{key} has high seed variability; explicit review is required")
        if aggregate.get("gain_limited") is True and not allow_gain_limited:
            raise ValueError(f"{key} gain hits a policy bound; explicit review is required")
        if aggregate.get("peak_safety_review_required") is True and not allow_peak_risk:
            raise ValueError(f"{key} positive gain predicts peak risk; explicit review is required")
        gain = _finite(aggregate.get("gain_db"), f"{key}.gain_db")
        measured_lufs = _finite(aggregate.get("median_lufs"), f"{key}.median_lufs")
        mad_lu = _finite(aggregate.get("mad_lu"), f"{key}.mad_lu")
        samples = aggregate.get("sample_count")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples != 9:
            raise ValueError(f"{key} must contain exactly nine measurements")
        counts[status] += 1
        voices[key] = {
            "gain_db": gain,
            "measured_lufs": measured_lufs,
            "reference_lufs": float(policy["reference_lufs"]),
            "mad_lu": mad_lu,
            "samples": samples,
            "method": "bs1770",
            "corpus_version": str(policy["name"]),
        }
    if not voices:
        raise ValueError("no eligible voice measurements to promote")
    candidate = {
        "schema": 1,
        "method": "bs1770",
        "corpus": str(policy["name"]),
        "reference_lufs": float(policy["reference_lufs"]),
        "generated_with": dict(sorted(generated_with.items())),
        "voices": dict(sorted(voices.items())),
    }
    return candidate, counts


def build_candidate_from_reports(
    reports: Sequence[Mapping[str, Any]],
    **options: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Merge candidates from disjoint complete reports with matching provenance."""
    if not reports:
        raise ValueError("at least one measurement report is required")
    merged: dict[str, Any] | None = None
    counts = {"eligible": 0, "high_variability": 0}
    for report in reports:
        candidate, next_counts = build_candidate(report, **options)
        if merged is None:
            merged = candidate
        else:
            for field in ("schema", "method", "corpus", "reference_lufs", "generated_with"):
                if candidate[field] != merged[field]:
                    raise ValueError(f"measurement reports have mismatched {field}")
            duplicates = set(merged["voices"]) & set(candidate["voices"])
            if duplicates:
                raise ValueError(
                    f"duplicate calibration key across reports: {sorted(duplicates)[0]}"
                )
            merged["voices"].update(candidate["voices"])
        for name, count in next_counts.items():
            counts[name] += count
    assert merged is not None
    merged["voices"] = dict(sorted(merged["voices"].items()))
    return merged, counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-high-variability", action="store_true")
    parser.add_argument("--allow-peak-risk", action="store_true")
    parser.add_argument("--allow-gain-limited", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_paths = args.reports or [DEFAULT_INPUT]
    output = args.output.resolve()
    if output == PRODUCTION_CATALOG.resolve():
        raise SystemExit("promotion never writes directly to the packaged production catalog")
    if any(output == report.resolve() for report in report_paths):
        raise SystemExit("candidate output must not overwrite a measurement report")
    if output.exists() and not args.force:
        raise SystemExit(f"output already exists: {output}; pass --force to replace it")
    try:
        reports = []
        for path in report_paths:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError(f"measurement report must be an object: {path}")
            reports.append(raw)
        candidate, counts = build_candidate_from_reports(
            reports,
            include_high_variability=args.include_high_variability,
            allow_peak_risk=args.allow_peak_risk,
            allow_gain_limited=args.allow_gain_limited,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"cannot build calibration candidate: {error}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Eligible: {counts['eligible']}")
    print(f"High variability: {counts['high_variability']}")
    print(f"Candidate: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
