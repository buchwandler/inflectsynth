#!/usr/bin/env python3
"""Measure deterministic prepared Inflect speech for static voice-level calibration."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from audiosig import measure_loudness

from inflectsynth import (
    InflectVoice,
    SynthesisConfig,
    VoiceCalibrationKey,
    VoiceLevelConfig,
    discover_models,
)

POLICY_PATH = Path(__file__).with_name("data") / "voice_level_policy.json"
STIMULI_PATH = Path(__file__).with_name("data") / "voice_level_stimuli.json"
DEFAULT_OUTPUT = Path("benchmarks/output/voice_level_calibration/measurements.json")
EXPECTED_MODELS = ("nano-v2", "micro-v2")
_POLICY_FIELDS = {
    "schema",
    "name",
    "seeds",
    "reference_lufs",
    "min_gain_db",
    "max_gain_db",
    "max_mad_lu",
    "identity_overrides",
}
_STIMULI_FIELDS = {"schema", "name", "language", "stimuli"}


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_pairs,
        parse_constant=_reject_constant,
    )


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Load and validate deterministic seed coverage and gain bounds."""
    policy = _read_json(path)
    if not isinstance(policy, Mapping):
        raise ValueError("voice-level policy must be an object")
    unknown = set(policy) - _POLICY_FIELDS
    if unknown or set(policy) != _POLICY_FIELDS:
        raise ValueError(f"voice-level policy has invalid fields: {sorted(unknown)}")
    if policy.get("schema") != 1 or isinstance(policy.get("schema"), bool):
        raise ValueError("voice-level policy has an unsupported schema")
    if policy.get("name") != "inflectsynth-prepared-speech-v1":
        raise ValueError("voice-level policy has an unsupported corpus name")
    seeds = policy.get("seeds")
    if (
        not isinstance(seeds, list)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or seeds != [0, 1, 2]
    ):
        raise ValueError("voice-level policy seeds must be exactly [0, 1, 2]")
    reference = _finite(policy.get("reference_lufs"), "reference_lufs")
    minimum = _finite(policy.get("min_gain_db"), "min_gain_db")
    maximum = _finite(policy.get("max_gain_db"), "max_gain_db")
    max_mad = _finite(policy.get("max_mad_lu"), "max_mad_lu")
    if minimum > maximum or max_mad < 0:
        raise ValueError("voice-level policy gain/variability limits are invalid")
    overrides = policy.get("identity_overrides")
    if not isinstance(overrides, Mapping):
        raise ValueError("identity_overrides must be an object")
    for key_text, override in overrides.items():
        if not isinstance(key_text, str) or not isinstance(override, Mapping):
            raise ValueError("identity overrides must map keys to objects")
        if set(override) - {"min_gain_db", "rationale"}:
            raise ValueError(f"identity override for {key_text} has unknown fields")
        VoiceCalibrationKey.parse(key_text)
        if _finite(override.get("min_gain_db"), f"{key_text}.min_gain_db") > maximum:
            raise ValueError(f"identity override for {key_text} exceeds max_gain_db")
        rationale = override.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"identity override for {key_text} needs a rationale")
    return {
        **dict(policy),
        "reference_lufs": reference,
        "min_gain_db": minimum,
        "max_gain_db": maximum,
        "max_mad_lu": max_mad,
    }


def load_stimuli(path: Path = STIMULI_PATH) -> dict[str, Any]:
    """Load prepared English stimuli with unique IDs and no implicit text shaping."""
    document = _read_json(path)
    if not isinstance(document, Mapping) or set(document) != _STIMULI_FIELDS:
        raise ValueError("stimuli document has an invalid shape")
    if document.get("schema") != 1 or isinstance(document.get("schema"), bool):
        raise ValueError("stimuli document has an unsupported schema")
    for name in ("name", "language"):
        value = document.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"stimuli {name} must be a non-empty string")
    stimuli = document.get("stimuli")
    if not isinstance(stimuli, list) or not stimuli:
        raise ValueError("stimuli must be a non-empty list")
    validated: list[dict[str, str]] = []
    seen: set[str] = set()
    for stimulus in stimuli:
        if not isinstance(stimulus, Mapping) or set(stimulus) != {"id", "text"}:
            raise ValueError("each stimulus must contain only id and text")
        stimulus_id, text = stimulus.get("id"), stimulus.get("text")
        if not isinstance(stimulus_id, str) or not stimulus_id.strip() or stimulus_id in seen:
            raise ValueError("stimulus IDs must be unique non-empty strings")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"stimulus {stimulus_id} must contain prepared speakable text")
        seen.add(stimulus_id)
        validated.append({"id": stimulus_id, "text": text})
    return {
        "schema": 1,
        "name": document["name"],
        "language": document["language"],
        "stimuli": validated,
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def generated_with() -> dict[str, str]:
    return {
        name: _package_version(name)
        for name in ("inflectsynth", "inflectg2p", "onnxvoice", "audiosig")
    }


def expand_identities(model: Any, *, requested_model: str) -> list[dict[str, Any]]:
    """Resolve one managed base model to stable Inflect calibration identities."""
    model_id = getattr(model, "model_id", None)
    model_ref = getattr(model, "model_ref", None)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("opened Inflect model has no stable model ID")
    if model_id != requested_model:
        raise ValueError(
            f"requested model {requested_model!r} resolved to unexpected ID {model_id!r}"
        )
    voices = getattr(model, "available_voices", ())
    if "default" not in voices:
        raise ValueError(f"managed model {model_id!r} does not expose the fixed 'default' voice")
    key = VoiceCalibrationKey("inflect", model_id, "default")
    return [
        {
            "model_source": "inflect",
            "requested_model": requested_model,
            "model_id": model_id,
            "model_ref": model_ref,
            "voice": "default",
            "calibration_key": str(key),
        }
    ]


class _SynthesisModel(Protocol):
    def synthesize_prepared(
        self,
        text: str,
        *,
        voice: str,
        config: SynthesisConfig,
    ) -> Any: ...


def _failure(
    entry: Mapping[str, Any], stimulus_id: str, seed: int, error: Exception, *, phase: str
) -> dict[str, Any]:
    return {
        **dict(entry),
        "stimulus_id": stimulus_id,
        "seed": seed,
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def measure_identities(
    model: _SynthesisModel,
    entries: Sequence[Mapping[str, Any]],
    stimuli: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure each stimulus using each declared seed with calibration disabled."""
    measurements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seeds = list(policy["seeds"])
    total = len(entries) * len(stimuli) * len(seeds)
    completed = 0
    for entry in entries:
        for stimulus in stimuli:
            for seed in seeds:
                completed += 1
                print(
                    f"[{completed}/{total}] {entry['calibration_key']} "
                    f"{stimulus['id']} seed {seed}",
                    flush=True,
                )
                config = SynthesisConfig(
                    speed=1.0,
                    variation=0.667,
                    seed=seed,
                    voice_level=VoiceLevelConfig(mode="off"),
                )
                try:
                    result = model.synthesize_prepared(
                        stimulus["text"],
                        voice=str(entry["voice"]),
                        config=config,
                    )
                except Exception as error:
                    failures.append(_failure(entry, stimulus["id"], seed, error, phase="synthesis"))
                    continue
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
                    loudness = measure_loudness(audio, sample_rate=sample_rate)
                    measured_lufs = loudness.integrated_lufs
                    if measured_lufs is None:
                        raise ValueError("integrated loudness is unavailable")
                    integrated_lufs = _finite(measured_lufs, "integrated_lufs")
                    peak = float(np.max(np.abs(audio)))
                    if not math.isfinite(peak):
                        raise ValueError("sample peak is not finite")
                    duration = float(result.duration_seconds)
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError("duration must be finite and positive")
                except Exception as error:
                    failures.append(
                        _failure(entry, stimulus["id"], seed, error, phase="measurement")
                    )
                    continue
                measurements.append(
                    {
                        **dict(entry),
                        "stimulus_id": stimulus["id"],
                        "seed": seed,
                        "requested_speed": 1.0,
                        "variation": 0.667,
                        "sample_rate": sample_rate,
                        "duration_seconds": duration,
                        "integrated_lufs": integrated_lufs,
                        "raw_peak": peak,
                        "calibration_mode": "off",
                    }
                )
    return measurements, failures


def _gain_for(median_lufs: float, key: str, policy: Mapping[str, Any]) -> tuple[float, bool]:
    override = policy["identity_overrides"].get(key)
    minimum = (
        float(override["min_gain_db"]) if override is not None else float(policy["min_gain_db"])
    )
    maximum = float(policy["max_gain_db"])
    requested = float(policy["reference_lufs"]) - median_lufs
    gain = min(maximum, max(minimum, requested))
    limited = math.isclose(gain, minimum, abs_tol=1e-12) or math.isclose(
        gain, maximum, abs_tol=1e-12
    )
    return gain, limited


def aggregate_measurements(
    entries: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    stimuli: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate seeds within stimuli, then aggregate stimulus medians."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in measurements:
        grouped[str(row["calibration_key"])].append(row)
    seeds = set(policy["seeds"])
    stimulus_ids = [str(stimulus["id"]) for stimulus in stimuli]
    aggregates: list[dict[str, Any]] = []
    for entry in entries:
        key = str(entry["calibration_key"])
        rows = grouped.get(key, [])
        by_stimulus: dict[str, dict[int, float]] = defaultdict(dict)
        duplicate = False
        for row in rows:
            stimulus_id = str(row.get("stimulus_id"))
            seed = row.get("seed")
            if (
                stimulus_id not in stimulus_ids
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed not in seeds
                or seed in by_stimulus[stimulus_id]
            ):
                duplicate = True
                continue
            by_stimulus[stimulus_id][seed] = _finite(row.get("integrated_lufs"), "integrated_lufs")
        medians: dict[str, float] = {}
        seed_mads: dict[str, float] = {}
        complete = not duplicate
        for stimulus_id in stimulus_ids:
            values_by_seed = by_stimulus.get(stimulus_id, {})
            if set(values_by_seed) != seeds:
                complete = False
                continue
            values = list(values_by_seed.values())
            median = statistics.median(values)
            medians[stimulus_id] = median
            seed_mads[stimulus_id] = statistics.median(abs(value - median) for value in values)
        if set(medians) != set(stimulus_ids):
            complete = False
        median_lufs = statistics.median(medians.values()) if complete else None
        mad_lu = max(seed_mads.values(), default=0.0)
        spread = max(medians.values()) - min(medians.values()) if medians else 0.0
        gain_db, limited = (
            _gain_for(median_lufs, key, policy) if median_lufs is not None else (None, False)
        )
        scale = 10.0 ** (float(gain_db) / 20.0) if gain_db is not None else 1.0
        predicted_peaks = [_finite(row.get("raw_peak"), "raw_peak") * scale for row in rows]
        predicted_peak = max(predicted_peaks, default=0.0)
        aggregates.append(
            {
                **dict(entry),
                "median_lufs": median_lufs,
                "mad_lu": mad_lu,
                "seed_mad_by_stimulus": seed_mads,
                "stimulus_medians": medians,
                "stimulus_spread_lu": spread,
                "sample_count": len(rows),
                "stimulus_count": len(medians),
                "status": (
                    "incomplete"
                    if not complete
                    else "high_variability"
                    if mad_lu > float(policy["max_mad_lu"])
                    else "eligible"
                ),
                "gain_db": gain_db,
                "gain_limited": limited,
                "max_raw_peak": max(
                    (_finite(row.get("raw_peak"), "raw_peak") for row in rows), default=0.0
                ),
                "predicted_peak": predicted_peak,
                "peak_safety_review_required": bool(
                    gain_db is not None and gain_db > 0 and predicted_peak > 1.0
                ),
                "complete": complete,
            }
        )
    return aggregates


def build_report(
    entries: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    *,
    stimuli: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
    language: str = "en-US",
) -> dict[str, Any]:
    aggregates = aggregate_measurements(entries, measurements, stimuli, policy)
    complete_identities = sum(row["complete"] is True for row in aggregates)
    expected_samples = len(entries) * len(stimuli) * len(policy["seeds"])
    coverage_complete = (
        complete_identities == len(entries)
        and len(measurements) == expected_samples
        and not failures
    )
    return {
        "schema": 1,
        "corpus": str(policy["name"]),
        "language": language,
        "generated_with": generated_with(),
        "policy": dict(policy),
        "coverage": {
            "identities_expected": len(entries),
            "identities_measured": complete_identities,
            "identities_failed": len(entries) - complete_identities,
            "measurements_expected": expected_samples,
            "measurements_measured": len(measurements),
            "measurement_failures": len(failures),
            "complete": coverage_complete,
        },
        "stimuli": [dict(stimulus) for stimulus in stimuli],
        "measurements": [dict(row) for row in measurements],
        "failures": [dict(row) for row in failures],
        "aggregates": aggregates,
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically write a deterministic, finite JSON evidence report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _open_failure(model_id: str, error: Exception) -> dict[str, Any]:
    key = str(VoiceCalibrationKey("inflect", model_id, "default"))
    return {
        "model_source": "inflect",
        "requested_model": model_id,
        "model_id": model_id,
        "model_ref": f"inflect:{model_id}",
        "voice": "default",
        "calibration_key": key,
        "phase": "model_open",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=EXPECTED_MODELS)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stimuli", type=Path, default=STIMULI_PATH)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policy = load_policy(args.policy)
    stimulus_document = load_stimuli(args.stimuli)
    if args.list_only:
        for discovered in discover_models(
            offline=args.offline,
            refresh=args.refresh_catalog,
            cache_dir=args.cache_dir,
        ):
            if discovered.id in EXPECTED_MODELS:
                print(f"{discovered.id}\t{discovered.default_voice}\t{discovered.sample_rate}")
        return 0

    requested_models = args.model or list(EXPECTED_MODELS)
    entries: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_id in requested_models:
        try:
            model = InflectVoice.from_pretrained(
                model_id,
                cache_dir=args.cache_dir,
                offline=args.offline,
                refresh_catalog=args.refresh_catalog,
            )
        except Exception as error:
            entries.append(
                {
                    "model_source": "inflect",
                    "requested_model": model_id,
                    "model_id": model_id,
                    "model_ref": f"inflect:{model_id}",
                    "voice": "default",
                    "calibration_key": str(VoiceCalibrationKey("inflect", model_id, "default")),
                }
            )
            failures.append(_open_failure(model_id, error))
            continue
        with model:
            model_entries = expand_identities(model, requested_model=model_id)
            entries.extend(model_entries)
            model_measurements, model_failures = measure_identities(
                model,
                model_entries,
                stimulus_document["stimuli"],
                policy,
            )
            measurements.extend(model_measurements)
            failures.extend(model_failures)

    report = build_report(
        entries,
        measurements,
        failures,
        stimuli=stimulus_document["stimuli"],
        policy=policy,
        language=stimulus_document["language"],
    )
    write_report(args.output, report)
    coverage = report["coverage"]
    print(f"Report: {args.output}")
    print(
        f"Coverage: {coverage['measurements_measured']}/{coverage['measurements_expected']} "
        f"measurements; {coverage['measurement_failures']} failure(s)"
    )
    return 0 if coverage["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
