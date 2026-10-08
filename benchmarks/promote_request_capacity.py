#!/usr/bin/env python3
"""Validate real request-capacity evidence and write a candidate policy only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import request_capacity_benchmark as benchmark

PRODUCTION_POLICY = (
    Path(__file__).resolve().parents[1] / "inflectsynth" / "data" / "request_capacity.json"
)
DEFAULT_REPORT = Path("benchmarks/output/request_capacity/report.json")
DEFAULT_OUTPUT = Path("benchmarks/output/request_capacity/candidate_policy.json")
_RUNTIME_IDENTITY_FIELDS = {
    "engine",
    "engine_version",
    "g2p_revision",
    "runtime_revision",
    "request_api_version",
}


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_provenance(
    report: Mapping[str, Any], expected_models: Sequence[str]
) -> list[Mapping[str, Any]]:
    generated_with = report.get("generated_with")
    required_versions = {"inflectsynth", "inflectg2p", "onnxvoice"}
    if (
        not isinstance(generated_with, Mapping)
        or set(generated_with) != required_versions
        or any(
            not isinstance(value, str) or not value.strip() or value == "unknown"
            for value in generated_with.values()
        )
    ):
        raise ValueError("report package-version provenance is incomplete")
    environment = report.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != {"python_version", "platform"}
        or any(not isinstance(value, str) or not value for value in environment.values())
    ):
        raise ValueError("report execution environment provenance is incomplete")
    if "android" in environment["platform"].casefold():
        raise ValueError("unsupported Android runtime cannot promote a production capacity policy")
    runtime = report.get("runtime_identity")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != _RUNTIME_IDENTITY_FIELDS
        or runtime.get("engine") != "inflect"
        or any(
            not isinstance(runtime.get(key), str) or not runtime[key]
            for key in _RUNTIME_IDENTITY_FIELDS
        )
    ):
        raise ValueError("report runtime identity is incomplete")
    models = report.get("models")
    if not isinstance(models, list) or len(models) != len(expected_models):
        raise ValueError("report must include provenance for both canonical models")
    by_id: dict[str, Mapping[str, Any]] = {}
    for model in models:
        if not isinstance(model, Mapping):
            raise ValueError("model provenance records must be objects")
        model_id = model.get("model_id")
        if model_id not in expected_models or model_id in by_id:
            raise ValueError(f"model IDs must be canonical and unique: {model_id!r}")
        if model.get("model_ref") != f"inflect:{model_id}":
            raise ValueError(f"model {model_id!r} has an invalid managed model reference")
        revision = model.get("source_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"model {model_id!r} has no source revision")
        artifacts = model.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {"duration", "decode"}:
            raise ValueError(f"model {model_id!r} artifact provenance is incomplete")
        for name, artifact in artifacts.items():
            if not isinstance(artifact, Mapping):
                raise ValueError(f"model {model_id!r} {name} artifact record is invalid")
            size = artifact.get("bytes")
            digest = artifact.get("sha256")
            if type(size) is not int or size <= 0:
                raise ValueError(f"model {model_id!r} {name} artifact size is invalid")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"model {model_id!r} {name} artifact digest is invalid")
        by_id[model_id] = model
    if set(by_id) != set(expected_models):
        raise ValueError("report model provenance does not cover both canonical models")
    return [by_id[model_id] for model_id in expected_models]


def validate_report(
    report: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
    stimuli: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    policy = dict(policy or benchmark.load_policy())
    stimuli = dict(stimuli or benchmark.load_stimuli())
    if report.get("schema") != 1 or isinstance(report.get("schema"), bool):
        raise ValueError("unsupported request-capacity report schema")
    if report.get("unit") != "model_tokens":
        raise ValueError("report unit must be model_tokens")
    if report.get("corpus") != policy["corpus"] or report.get("policy") != policy:
        raise ValueError("report does not match the checked-in request-capacity policy")
    if report.get("stimuli") != stimuli:
        raise ValueError("report stimuli differ from the checked-in characterization corpus")
    model_provenance = _validate_provenance(report, policy["models"])
    if report.get("model_open_failures") != []:
        raise ValueError("report contains model-open failures")
    coverage = report.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("complete") is not True:
        raise ValueError("report coverage is incomplete")
    expected = len(policy["models"]) * len(policy["semantic_shapes"])
    expected *= len(policy["target_token_counts"]) * len(policy["seeds"])
    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != expected:
        raise ValueError("report sample count does not cover the required matrix")
    if (
        coverage.get("models_expected") != len(policy["models"])
        or coverage.get("models_opened") != len(policy["models"])
        or coverage.get("samples_expected") != expected
        or coverage.get("samples_recorded") != expected
        or coverage.get("synthesis_failures") != 0
    ):
        raise ValueError("report coverage counts do not match the complete matrix")

    keys: set[tuple[str, str, int, int]] = set()
    normalized_samples: list[Mapping[str, Any]] = []
    for row in samples:
        if not isinstance(row, Mapping):
            raise ValueError("sample rows must be objects")
        model_id = row.get("model")
        shape = row.get("semantic_shape")
        target = row.get("target_token_count")
        seed = row.get("seed")
        if model_id not in policy["models"]:
            raise ValueError(f"sample has a non-canonical model ID: {model_id!r}")
        if shape not in policy["semantic_shapes"] or target not in policy["target_token_counts"]:
            raise ValueError("sample references an unknown shape or target token neighborhood")
        if type(seed) is not int or seed not in policy["seeds"]:
            raise ValueError("sample seed is outside the declared deterministic seed set")
        key = (model_id, shape, target, seed)
        if key in keys:
            raise ValueError(f"duplicate request-capacity sample: {key}")
        keys.add(key)
        if row.get("synthesis_success") is not True or row.get("exception_type") is not None:
            raise ValueError(f"required sample {key} failed synthesis")
        token_count = row.get("token_count")
        if type(token_count) is not int or token_count <= 0:
            raise ValueError(f"sample {key} has an invalid model-token count")
        for field in ("source_text_length", "normalized_text_length", "audio_samples"):
            if type(row.get(field)) is not int or row[field] <= 0:
                raise ValueError(f"sample {key} has invalid {field}")
        if row.get("speed") != policy["speed"] or row.get("variation") != policy["variation"]:
            raise ValueError(f"sample {key} used unexpected synthesis controls")
        if row.get("sample_rate") != policy["sample_rate"] or isinstance(
            row.get("sample_rate"), bool
        ):
            raise ValueError(f"sample {key} has an unexpected sample rate")
        if row.get("finite_pcm") is not True:
            raise ValueError(f"sample {key} PCM is not finite")
        audio_seconds = _finite(row.get("audio_seconds"), f"{key}.audio_seconds")
        if audio_seconds <= 0 or audio_seconds > float(policy["max_audio_seconds"]):
            raise ValueError(f"sample {key} has an implausible output duration")
        if _finite(row.get("real_seconds"), f"{key}.real_seconds") < 0:
            raise ValueError(f"sample {key} has invalid runtime duration")
        if _finite(row.get("xrt"), f"{key}.xrt") <= 0:
            raise ValueError(f"sample {key} has invalid real-time factor")
        if _finite(row.get("peak_abs"), f"{key}.peak_abs") < 0:
            raise ValueError(f"sample {key} has invalid PCM peak")
        normalized_samples.append(row)
    if len(keys) != expected:
        raise ValueError("report does not cover every model/shape/target/seed combination")
    return policy, model_provenance, normalized_samples


def safe_observation_boundaries(
    samples: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, int]:
    boundaries: dict[str, int] = {}
    for model_id in policy["models"]:
        group_boundaries: list[int] = []
        for shape in policy["semantic_shapes"]:
            for seed in policy["seeds"]:
                observed = [
                    row["token_count"]
                    for row in samples
                    if row["model"] == model_id
                    and row["semantic_shape"] == shape
                    and row["seed"] == seed
                ]
                if not observed:
                    raise ValueError(f"no safe observations for {model_id}/{shape}/seed-{seed}")
                group_boundaries.append(max(observed))
        boundaries[model_id] = min(group_boundaries)
    return boundaries


def build_candidate(report: Mapping[str, Any], maxima: Mapping[str, int]) -> dict[str, Any]:
    policy, models, samples = validate_report(report)
    if set(maxima) != set(policy["models"]):
        raise ValueError("candidate maxima must be supplied for both canonical models")
    boundaries = safe_observation_boundaries(samples, policy)
    model_policy: dict[str, dict[str, Any]] = {}
    evidence_revision = _canonical_sha256(report)
    for model_id in policy["models"]:
        maximum = maxima[model_id]
        if type(maximum) is not int or maximum <= 0:
            raise ValueError(f"candidate maximum for {model_id} must be a positive integer")
        if maximum > boundaries[model_id]:
            raise ValueError(
                f"candidate maximum for {model_id} exceeds safe observation boundary "
                f"{boundaries[model_id]} model tokens"
            )
        model_policy[model_id] = {
            "maximum": maximum,
            "basis": "validated-safe-request-budget",
            "corpus": policy["corpus"],
            "safe_observation_boundary": boundaries[model_id],
        }
    payload: dict[str, Any] = {
        "schema": 1,
        "unit": "model_tokens",
        "corpus": policy["corpus"],
        "evidence_sha256": evidence_revision,
        "runtime_identity": dict(report["runtime_identity"]),
        "environment": dict(report["environment"]),
        "generated_with": dict(report["generated_with"]),
        "provenance": [dict(record) for record in models],
        "models": model_policy,
    }
    payload["revision"] = _canonical_sha256(payload)
    return payload


def _parse_maxima(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--maximum must use MODEL=TOKEN_COUNT")
        model_id, raw_count = value.split("=", 1)
        if model_id in result or model_id not in benchmark.EXPECTED_MODELS:
            raise ValueError(f"duplicate or non-canonical model ID in --maximum: {model_id!r}")
        try:
            count = int(raw_count)
        except ValueError:
            raise ValueError(f"invalid maximum for {model_id}: {raw_count!r}") from None
        if str(count) != raw_count:
            raise ValueError(f"maximum for {model_id} must be a plain positive integer")
        result[model_id] = count
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--maximum", action="append", required=True, metavar="MODEL=TOKEN_COUNT")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="replace an existing candidate output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output == PRODUCTION_POLICY.resolve():
        raise SystemExit(
            "promotion writes candidate policies only; it never overwrites production policy"
        )
    if output == report_path:
        raise SystemExit("candidate output must not overwrite the source benchmark report")
    if output.exists() and not args.force:
        raise SystemExit(f"candidate output already exists: {output}; pass --force to replace it")
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"), object_pairs_hook=_json_pairs)
        if not isinstance(raw, Mapping):
            raise ValueError("benchmark report must be an object")
        maxima = _parse_maxima(args.maximum)
        candidate = build_candidate(raw, maxima)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"cannot promote request capacity: {error}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Candidate policy: {output}")
    print(f"Policy revision: {candidate['revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
