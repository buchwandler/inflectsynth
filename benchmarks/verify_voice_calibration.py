#!/usr/bin/env python3
"""Verify packaged Inflect calibration with real post-gain synthesis."""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from audiosig import measure_loudness

from inflectsynth import (
    InflectVoice,
    SynthesisConfig,
    VoiceCalibrationKey,
    VoiceLevelConfig,
    load_voice_calibration,
)

from . import voice_level_benchmark as benchmark

POLICY_PATH = Path(__file__).with_name("data") / "voice_calibration_verification_policy.json"
DEFAULT_OUTPUT = Path("benchmarks/output/voice_level_calibration/verification.json")
CATALOG_RESOURCE = files("inflectsynth").joinpath("data").joinpath("voice_level_calibration.json")
EXPECTED_MODELS = ("nano-v2", "micro-v2")
EXPECTED_VOICES = ("default",)
EXPECTED_SEEDS = (0, 1, 2)
_POLICY_FIELDS = {
    "schema",
    "name",
    "reference_lufs",
    "max_post_calibration_abs_error_lu",
    "max_post_calibration_peak_dbfs",
    "models",
    "voices",
    "seeds",
}


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_verification_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Load the release verification scope and fixed acceptance limits."""
    policy = benchmark._read_json(path)
    if not isinstance(policy, Mapping) or set(policy) != _POLICY_FIELDS:
        raise ValueError("calibrated verification policy has an invalid shape")
    if policy.get("schema") != 1 or isinstance(policy.get("schema"), bool):
        raise ValueError("calibrated verification policy has an unsupported schema")
    if policy.get("name") != "inflectsynth-calibrated-verification-v1":
        raise ValueError("calibrated verification policy has an unsupported name")
    reference = _finite(policy.get("reference_lufs"), "reference_lufs")
    max_error = _finite(
        policy.get("max_post_calibration_abs_error_lu"),
        "max_post_calibration_abs_error_lu",
    )
    max_peak = _finite(
        policy.get("max_post_calibration_peak_dbfs"),
        "max_post_calibration_peak_dbfs",
    )
    if max_error < 0 or max_peak > 0:
        raise ValueError("calibrated verification limits are invalid")
    if policy.get("models") != list(EXPECTED_MODELS):
        raise ValueError("verification policy must cover Nano v2 and Micro v2")
    if policy.get("voices") != list(EXPECTED_VOICES):
        raise ValueError("verification policy must cover the fixed default voice")
    if policy.get("seeds") != list(EXPECTED_SEEDS):
        raise ValueError("verification policy seeds must be exactly [0, 1, 2]")
    return {
        **dict(policy),
        "reference_lufs": reference,
        "max_post_calibration_abs_error_lu": max_error,
        "max_post_calibration_peak_dbfs": max_peak,
    }


def _identities(policy: Mapping[str, Any], catalog: Any) -> list[dict[str, Any]]:
    rows = []
    for model_id in policy["models"]:
        for voice in policy["voices"]:
            key = VoiceCalibrationKey("inflect", model_id, voice)
            calibration = catalog.voices.get(key)
            rows.append(
                {
                    "model_id": model_id,
                    "voice": voice,
                    "calibration_key": str(key),
                    "expected_gain_db": calibration.gain_db if calibration else None,
                    "status": "pending",
                    "complete": False,
                    "stimulus_medians_lufs": {},
                    "median_lufs": None,
                    "abs_error_lu": None,
                    "max_sample_peak_dbfs": None,
                    "measurements": [],
                    "issues": [],
                }
            )
    return rows


def _add_issue(
    identity: dict[str, Any],
    failures: list[dict[str, Any]],
    *,
    phase: str,
    message: str,
    error_type: str = "VerificationFailure",
    stimulus_id: str | None = None,
    seed: int | None = None,
) -> None:
    issue = {
        "model_id": identity["model_id"],
        "voice": identity["voice"],
        "calibration_key": identity["calibration_key"],
        "phase": phase,
        "error_type": error_type,
        "error": message,
    }
    if stimulus_id is not None:
        issue["stimulus_id"] = stimulus_id
    if seed is not None:
        issue["seed"] = seed
    failures.append(issue)
    identity["issues"].append(message)


def _check_runtime_metadata(
    result: Any,
    identity: Mapping[str, Any],
    catalog_revision: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None, ["synthesis result metadata is missing"]
    voice_level = metadata.get("voice_level")
    if not isinstance(voice_level, Mapping):
        return None, ["synthesis result voice_level metadata is missing"]
    expected_fields = {
        "mode": "calibrated",
        "source": "catalog",
        "calibration_key": identity["calibration_key"],
        "catalog_revision": catalog_revision,
    }
    issues = [
        f"voice_level.{name} expected {expected!r}, got {voice_level.get(name)!r}"
        for name, expected in expected_fields.items()
        if voice_level.get(name) != expected
    ]
    try:
        gain = _finite(voice_level.get("gain_db"), "voice_level.gain_db")
    except ValueError:
        gain = None
        issues.append("voice_level.gain_db is missing or non-finite")
    expected_gain = identity["expected_gain_db"]
    if gain is not None and not math.isclose(
        gain, float(expected_gain), rel_tol=0.0, abs_tol=1e-12
    ):
        issues.append(f"voice_level.gain_db expected {expected_gain!r}, got {gain!r}")
    return {
        "mode": voice_level.get("mode"),
        "source": voice_level.get("source"),
        "calibration_key": voice_level.get("calibration_key"),
        "catalog_revision": voice_level.get("catalog_revision"),
        "gain_db": gain,
    }, issues


def _measure_identity(
    model: Any,
    identity: dict[str, Any],
    stimuli: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
    catalog_revision: str,
    failures: list[dict[str, Any]],
) -> None:

    values_by_stimulus: dict[str, list[float]] = defaultdict(list)
    peaks_dbfs: list[float] = []
    config = SynthesisConfig(
        speed=1.0,
        variation=0.667,
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )
    for stimulus in stimuli:
        for seed in policy["seeds"]:
            row: dict[str, Any] = {
                "stimulus_id": stimulus["id"],
                "seed": seed,
                "status": "failed",
            }
            try:
                result = model.synthesize_prepared(
                    stimulus["text"],
                    voice=identity["voice"],
                    config=SynthesisConfig(
                        speed=config.speed,
                        variation=config.variation,
                        seed=seed,
                        voice_level=config.voice_level,
                    ),
                )
            except Exception as error:
                row.update(
                    {"phase": "synthesis", "error_type": type(error).__name__, "error": str(error)}
                )
                _add_issue(
                    identity,
                    failures,
                    phase="synthesis",
                    message=str(error),
                    error_type=type(error).__name__,
                    stimulus_id=stimulus["id"],
                    seed=seed,
                )
                identity["measurements"].append(row)
                continue

            metadata, metadata_issues = _check_runtime_metadata(result, identity, catalog_revision)
            row["voice_level"] = metadata
            for message in metadata_issues:
                _add_issue(
                    identity,
                    failures,
                    phase="metadata",
                    message=message,
                    stimulus_id=stimulus["id"],
                    seed=seed,
                )
            try:
                audio = np.asarray(result.audio, dtype=np.float32)
                if audio.ndim != 1 or not audio.size or not np.all(np.isfinite(audio)):
                    raise ValueError("synthesis audio must be non-empty, finite, and mono")
                sample_rate = result.sample_rate
                if (
                    isinstance(sample_rate, bool)
                    or not isinstance(sample_rate, int)
                    or sample_rate <= 0
                ):
                    raise ValueError("sample rate must be a positive integer")
                peak = float(np.max(np.abs(audio)))
                if not math.isfinite(peak) or peak <= 0:
                    raise ValueError("sample peak must be positive and finite")
                peak_dbfs = 20.0 * math.log10(peak)
                loudness = measure_loudness(audio, sample_rate=sample_rate)
                integrated_lufs = loudness.integrated_lufs
                if integrated_lufs is None:
                    raise ValueError("integrated loudness is unavailable")
                integrated_lufs = _finite(integrated_lufs, "integrated_lufs")
            except Exception as error:
                row.update(
                    {
                        "phase": "measurement",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                _add_issue(
                    identity,
                    failures,
                    phase="measurement",
                    message=str(error),
                    error_type=type(error).__name__,
                    stimulus_id=stimulus["id"],
                    seed=seed,
                )
                identity["measurements"].append(row)
                continue
            row.update(
                {
                    "sample_rate": sample_rate,
                    "sample_peak": peak,
                    "sample_peak_dbfs": peak_dbfs,
                    "integrated_lufs": integrated_lufs,
                    "status": "passed" if not metadata_issues else "failed",
                }
            )

            values_by_stimulus[stimulus["id"]].append(integrated_lufs)
            peaks_dbfs.append(peak_dbfs)
            identity["measurements"].append(row)

    expected_count = len(stimuli) * len(policy["seeds"])
    identity["complete"] = len(identity["measurements"]) == expected_count and all(
        len(values_by_stimulus[stimulus["id"]]) == len(policy["seeds"]) for stimulus in stimuli
    )
    if identity["complete"]:
        medians = {
            stimulus["id"]: statistics.median(values_by_stimulus[stimulus["id"]])
            for stimulus in stimuli
        }
        median_lufs = statistics.median(medians.values())
        abs_error = abs(median_lufs - float(policy["reference_lufs"]))
        identity["stimulus_medians_lufs"] = medians
        identity["median_lufs"] = median_lufs
        identity["abs_error_lu"] = abs_error
        if abs_error > float(policy["max_post_calibration_abs_error_lu"]) + 1e-9:
            _add_issue(
                identity,
                failures,
                phase="acceptance",
                message=(
                    f"absolute median loudness error {abs_error:.6f} LU exceeds "
                    f"{policy['max_post_calibration_abs_error_lu']:.6f} LU"
                ),
            )
    else:
        _add_issue(
            identity,
            failures,
            phase="coverage",
            message=(
                "identity has incomplete measurements: "
                f"{len(identity['measurements'])} of {expected_count}"
            ),
        )
    if peaks_dbfs:
        peak = max(peaks_dbfs)
        identity["max_sample_peak_dbfs"] = peak
        if peak > float(policy["max_post_calibration_peak_dbfs"]) + 1e-9:
            _add_issue(
                identity,
                failures,
                phase="acceptance",
                message=(
                    f"maximum sample peak {peak:.6f} dBFS exceeds "
                    f"{policy['max_post_calibration_peak_dbfs']:.6f} dBFS"
                ),
            )
    identity["status"] = "passed" if identity["complete"] and not identity["issues"] else "failed"


def run_verification(
    *,
    open_model: Callable[..., Any] | None = None,
    cache_dir: Path | None = None,
    offline: bool = False,
    refresh_catalog: bool = False,
    policy_path: Path = POLICY_PATH,
    catalog: Any | None = None,
) -> dict[str, Any]:
    """Run calibrated post-gain loudness, runtime metadata, and peak checks."""
    policy = load_verification_policy(policy_path)
    measurement_policy = benchmark.load_policy()
    if policy["reference_lufs"] != measurement_policy["reference_lufs"]:
        raise ValueError("verification and calibration policies use different reference LUFS")
    stimuli_document = benchmark.load_stimuli()
    selected_catalog = catalog if catalog is not None else load_voice_calibration(CATALOG_RESOURCE)
    catalog_bytes = CATALOG_RESOURCE.read_bytes() if catalog is None else b"injected-test-catalog"
    identities = _identities(policy, selected_catalog)
    failures: list[dict[str, Any]] = []
    for identity in identities:
        if identity["expected_gain_db"] is None:
            _add_issue(
                identity,
                failures,
                phase="catalog",
                message="packaged catalog has no exact entry for this managed identity",
            )

    factory = open_model or InflectVoice.from_pretrained
    for model_id in policy["models"]:
        model_id_identities = [row for row in identities if row["model_id"] == model_id]
        active = [row for row in model_id_identities if row["expected_gain_db"] is not None]
        if not active:
            continue
        try:
            model = factory(
                model_id,
                cache_dir=cache_dir,
                offline=offline,
                refresh_catalog=refresh_catalog,
            )
        except Exception as error:
            for identity in active:
                _add_issue(
                    identity,
                    failures,
                    phase="model_open",
                    message=str(error),
                    error_type=type(error).__name__,
                )
            continue
        try:
            if getattr(model, "model_id", None) != model_id:
                for identity in active:
                    _add_issue(
                        identity,
                        failures,
                        phase="model_identity",
                        message=f"opened model ID does not match {model_id!r}",
                    )
                continue
            available = set(getattr(model, "available_voices", ()))
            for identity in active:
                if identity["voice"] not in available:
                    _add_issue(
                        identity,
                        failures,
                        phase="voice_identity",
                        message=f"expected fixed voice {identity['voice']!r} is unavailable",
                    )
                    continue
                actual_key = model.calibration_key(identity["voice"])
                if actual_key != VoiceCalibrationKey.parse(identity["calibration_key"]):
                    _add_issue(
                        identity,
                        failures,
                        phase="voice_identity",
                        message="opened model resolved to an unexpected calibration identity",
                    )
                    continue
                _measure_identity(
                    model,
                    identity,
                    stimuli_document["stimuli"],
                    policy,
                    str(selected_catalog.revision),
                    failures,
                )
        except Exception as error:
            for identity in active:
                if identity["status"] == "pending":
                    _add_issue(
                        identity,
                        failures,
                        phase="verification",
                        message=str(error),
                        error_type=type(error).__name__,
                    )
        finally:
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    for identity in active:
                        _add_issue(
                            identity,
                            failures,
                            phase="model_close",
                            message=str(error),
                            error_type=type(error).__name__,
                        )

    for identity in identities:
        if identity["status"] == "pending" and not identity["issues"]:
            _add_issue(
                identity, failures, phase="verification", message="identity was not verified"
            )
        identity["status"] = (
            "passed" if identity["complete"] and not identity["issues"] else "failed"
        )
    measured = sum(identity["complete"] is True for identity in identities)
    passed = sum(identity["status"] == "passed" for identity in identities)
    versions = benchmark.generated_with()
    versions["python"] = platform.python_version()
    return {
        "schema": 1,
        "status": "passed" if passed == len(identities) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verifier": {"name": "inflectsynth-calibrated-verification", "schema": 1},
        "generated_with": dict(sorted(versions.items())),
        "catalog": {
            "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "revision": selected_catalog.revision,
            "schema": selected_catalog.schema,
            "corpus": selected_catalog.corpus,
            "reference_lufs": selected_catalog.reference_lufs,
        },
        "policy": policy,
        "measurement_policy": {
            "schema": measurement_policy["schema"],
            "name": measurement_policy["name"],
            "seeds": measurement_policy["seeds"],
        },
        "stimuli": {
            "schema": stimuli_document["schema"],
            "name": stimuli_document["name"],
            "language": stimuli_document["language"],
            "sha256": hashlib.sha256(benchmark.STIMULI_PATH.read_bytes()).hexdigest(),
            "items": stimuli_document["stimuli"],
        },
        "coverage": {
            "identities_expected": len(identities),
            "identities_measured": measured,
            "identities_passed": passed,
            "identities_failed": len(identities) - passed,
            "measurement_failures": sum(
                failure["phase"] in {"synthesis", "measurement"} for failure in failures
            ),
            "failure_count": len(failures),
            "complete": measured == len(identities),
        },
        "identities": identities,
        "failures": failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_verification(
            cache_dir=args.cache_dir,
            offline=args.offline,
            refresh_catalog=args.refresh_catalog,
        )
    except Exception as error:
        print(f"Calibrated verification could not start: {type(error).__name__}: {error}")
        return 1
    benchmark.write_report(args.output, report)
    coverage = report["coverage"]
    print(f"Report: {args.output}")
    print(
        f"Coverage: {coverage['identities_passed']}/{coverage['identities_expected']} passed; "
        f"{coverage['measurement_failures']} failure(s)"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
