#!/usr/bin/env python3
"""Opt-in characterization of real InflectSynth request sizes and audio output."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from inflectsynth import (
    DEFAULT_VOICE,
    InflectVoice,
    RequestMeasure,
    SynthesisConfig,
    VoiceLevelConfig,
    runtime_identity,
)

POLICY_PATH = Path(__file__).with_name("data") / "request_capacity_policy.json"
STIMULI_PATH = Path(__file__).with_name("data") / "request_capacity_stimuli.json"
DEFAULT_OUTPUT = Path("benchmarks/output/request_capacity/report.json")
EXPECTED_MODELS = ("nano-v2", "micro-v2")
_POLICY_FIELDS = {
    "schema",
    "corpus",
    "models",
    "target_token_counts",
    "semantic_shapes",
    "seeds",
    "speed",
    "variation",
    "sample_rate",
    "max_audio_seconds",
}
_STIMULI_FIELDS = {"schema", "name", "language", "phrases"}


class _Measurer(Protocol):
    def measure_prepared(self, text: str) -> RequestMeasure: ...


class _BenchmarkModel(_Measurer, Protocol):
    model_id: str
    model_ref: str | None

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    @property
    def installed(self) -> Any: ...

    def synthesize_prepared(self, text: str, *, voice: str, config: SynthesisConfig) -> Any: ...


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_pairs)


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    policy = _read_json(path)
    if not isinstance(policy, Mapping) or set(policy) != _POLICY_FIELDS:
        raise ValueError("request-capacity policy has an invalid shape")
    if policy.get("schema") != 1 or isinstance(policy.get("schema"), bool):
        raise ValueError("request-capacity policy has an unsupported schema")
    if policy.get("corpus") != "inflectsynth-request-capacity-v1":
        raise ValueError("request-capacity policy has an unsupported corpus")
    models = policy.get("models")
    if models != list(EXPECTED_MODELS):
        raise ValueError(f"models must be the canonical IDs {list(EXPECTED_MODELS)}")
    targets = policy.get("target_token_counts")
    if (
        not isinstance(targets, list)
        or not targets
        or any(type(target) is not int or target <= 0 for target in targets)
        or targets != sorted(set(targets))
    ):
        raise ValueError("target_token_counts must be unique increasing positive integers")
    shapes = policy.get("semantic_shapes")
    if shapes != ["one_sentence", "multiple_sentences"]:
        raise ValueError("both one-sentence and multiple-sentence shapes are required")
    seeds = policy.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("at least three unique non-negative deterministic seeds are required")
    if policy.get("speed") != 1.0 or policy.get("variation") != 0.667:
        raise ValueError("the benchmark controls must be speed=1.0 and variation=0.667")
    if policy.get("sample_rate") != 24_000 or isinstance(policy.get("sample_rate"), bool):
        raise ValueError("the expected sample rate must be 24000 Hz")
    max_audio_seconds = policy.get("max_audio_seconds")
    if (
        isinstance(max_audio_seconds, bool)
        or not isinstance(max_audio_seconds, (int, float))
        or not math.isfinite(float(max_audio_seconds))
        or max_audio_seconds <= 0
    ):
        raise ValueError("max_audio_seconds must be positive and finite")
    return dict(policy)


def load_stimuli(path: Path = STIMULI_PATH) -> dict[str, Any]:
    document = _read_json(path)
    if not isinstance(document, Mapping) or set(document) != _STIMULI_FIELDS:
        raise ValueError("request-capacity stimuli have an invalid shape")
    if document.get("schema") != 1 or isinstance(document.get("schema"), bool):
        raise ValueError("request-capacity stimuli have an unsupported schema")
    for field in ("name", "language"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            raise ValueError(f"stimuli {field} must be a non-empty string")
    phrases = document.get("phrases")
    if not isinstance(phrases, list) or len(phrases) < 4:
        raise ValueError("stimuli must provide at least four phrase fragments")
    if any(not isinstance(item, str) or not item.strip() for item in phrases):
        raise ValueError("every stimulus phrase must be a non-empty string")
    return {**dict(document), "phrases": list(phrases)}


def _text_with_phrase_count(phrases: Sequence[str], count: int, shape: str) -> str:
    selected = [phrases[index % len(phrases)].strip().rstrip(".!?;:") for index in range(count)]
    if shape == "one_sentence":
        return ", and ".join(selected) + "."
    if shape != "multiple_sentences":
        raise ValueError(f"unsupported semantic shape: {shape!r}")
    return " ".join(f"{phrase}." for phrase in selected)


def choose_text_for_token_target(
    measurer: _Measurer,
    phrases: Sequence[str],
    target: int,
    shape: str,
) -> tuple[str, int]:
    """Choose a deterministic stimulus using frontend token counts, not character length."""
    if type(target) is not int or target <= 0:
        raise ValueError("target must be a positive integer")
    minimum_phrases = 1 if shape == "one_sentence" else 2
    maximum_phrases = max(minimum_phrases, target * 4)
    measured: dict[int, int] = {}

    def count_tokens(phrase_count: int) -> int:
        if phrase_count not in measured:
            text = _text_with_phrase_count(phrases, phrase_count, shape)
            measured[phrase_count] = measurer.measure_prepared(text).amount
        return measured[phrase_count]

    low = minimum_phrases
    high = low
    while count_tokens(high) < target and high < maximum_phrases:
        low = high
        high = min(maximum_phrases, high * 2)
    while low + 1 < high:
        middle = (low + high) // 2
        if count_tokens(middle) < target:
            low = middle
        else:
            high = middle
    candidate_counts = {minimum_phrases, low, high, min(maximum_phrases, high + 1)}
    best_count = min(
        candidate_counts,
        key=lambda amount: (abs(count_tokens(amount) - target), count_tokens(amount) > target),
    )
    text = _text_with_phrase_count(phrases, best_count, shape)
    return text, count_tokens(best_count)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("inflectsynth", "inflectg2p", "onnxvoice"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_provenance(model: _BenchmarkModel) -> dict[str, Any]:
    metadata = model.metadata
    upstream = metadata.get("upstream", {})
    upstream = upstream if isinstance(upstream, Mapping) else {}
    revision = (
        metadata.get("source_revision")
        or upstream.get("source_revision")
        or upstream.get("revision")
    )
    artifacts: dict[str, dict[str, Any]] = {}
    for name in ("duration", "decode"):
        path = Path(getattr(model.installed, f"{name}_path"))
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {
        "model_id": model.model_id,
        "model_ref": model.model_ref,
        "source_revision": revision,
        "artifacts": artifacts,
    }


def measure_sample(
    model: _BenchmarkModel,
    *,
    model_id: str,
    semantic_shape: str,
    target_token_count: int,
    text: str,
    seed: int,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model_id,
        "semantic_shape": semantic_shape,
        "target_token_count": target_token_count,
        "source_text_length": len(text),
        "normalized_text_length": None,
        "token_count": None,
        "seed": seed,
        "speed": float(policy["speed"]),
        "variation": float(policy["variation"]),
        "synthesis_success": False,
        "exception_type": None,
        "audio_samples": None,
        "audio_seconds": None,
        "real_seconds": None,
        "xrt": None,
        "sample_rate": None,
        "finite_pcm": None,
        "peak_abs": None,
    }
    started = time.perf_counter()
    phase = "measurement"
    try:
        measurement = model.measure_prepared(text)
        row["token_count"] = measurement.amount
        phase = "synthesis"
        config = SynthesisConfig(
            speed=float(policy["speed"]),
            variation=float(policy["variation"]),
            seed=seed,
            voice_level=VoiceLevelConfig(mode="off"),
        )
        result = model.synthesize_prepared(text, voice=DEFAULT_VOICE, config=config)
        row["normalized_text_length"] = len(result.metadata["normalized_text"])
        if result.metadata["token_count"] != measurement.amount:
            raise ValueError("measurement token count differs from synthesis token count")
        audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        finite_pcm = bool(audio.size > 0 and np.all(np.isfinite(audio)))
        row["audio_samples"] = int(audio.size)
        row["sample_rate"] = int(result.sample_rate)
        row["audio_seconds"] = float(audio.size / result.sample_rate)
        row["finite_pcm"] = finite_pcm
        row["peak_abs"] = float(np.max(np.abs(audio))) if audio.size else None
        if not finite_pcm:
            raise ValueError("audio must be non-empty and finite")
        row["synthesis_success"] = True
    except Exception as error:
        row["exception_type"] = type(error).__name__
        row["exception_phase"] = phase
    elapsed = time.perf_counter() - started
    row["real_seconds"] = elapsed
    audio_seconds = row["audio_seconds"]
    row["xrt"] = (
        elapsed / audio_seconds if isinstance(audio_seconds, float) and audio_seconds > 0 else None
    )
    return row


def run_model(
    model: _BenchmarkModel,
    *,
    model_id: str,
    policy: Mapping[str, Any],
    stimuli: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if model.model_id != model_id:
        raise ValueError(f"requested model {model_id!r} resolved to {model.model_id!r}")
    rows: list[dict[str, Any]] = []
    for shape in policy["semantic_shapes"]:
        for target in policy["target_token_counts"]:
            text, _actual_tokens = choose_text_for_token_target(
                model, stimuli["phrases"], target, shape
            )
            for seed in policy["seeds"]:
                rows.append(
                    measure_sample(
                        model,
                        model_id=model_id,
                        semantic_shape=shape,
                        target_token_count=target,
                        text=text,
                        seed=seed,
                        policy=policy,
                    )
                )
    return rows


def build_report(
    policy: Mapping[str, Any],
    stimuli: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    model_open_failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = len(policy["models"]) * len(policy["semantic_shapes"])
    expected *= len(policy["target_token_counts"]) * len(policy["seeds"])
    complete = (
        len(samples) == expected
        and not model_open_failures
        and all(row.get("synthesis_success") is True for row in samples)
    )
    return {
        "schema": 1,
        "unit": "model_tokens",
        "corpus": policy["corpus"],
        "policy": dict(policy),
        "stimuli": dict(stimuli),
        "generated_with": _package_versions(),
        "runtime_identity": dict(runtime_identity()),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "models": [dict(item) for item in models],
        "model_open_failures": [dict(item) for item in model_open_failures],
        "coverage": {
            "models_expected": len(policy["models"]),
            "models_opened": len(models),
            "samples_expected": expected,
            "samples_recorded": len(samples),
            "synthesis_failures": sum(row.get("synthesis_success") is not True for row in samples),
            "complete": complete,
        },
        "samples": [dict(row) for row in samples],
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-real-models",
        action="store_true",
        help="explicitly opt in to loading/running real ONNX models",
    )
    parser.add_argument("--model", action="append", choices=EXPECTED_MODELS)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--stimuli", type=Path, default=STIMULI_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.run_real_models:
        raise SystemExit("real-model benchmark is gated; pass --run-real-models explicitly")
    policy = load_policy(args.policy)
    stimuli = load_stimuli(args.stimuli)
    requested = args.model or list(policy["models"])
    models: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    open_failures: list[dict[str, Any]] = []
    for model_id in requested:
        try:
            voice = InflectVoice.from_pretrained(
                model_id, cache_dir=args.cache_dir, offline=args.offline
            )
        except Exception as error:
            open_failures.append(
                {"model": model_id, "exception_type": type(error).__name__, "error": str(error)}
            )
            continue
        with voice:
            models.append(model_provenance(voice))
            samples.extend(run_model(voice, model_id=model_id, policy=policy, stimuli=stimuli))
    report = build_report(policy, stimuli, models, samples, open_failures)
    write_report(args.output, report)
    coverage = report["coverage"]
    print(f"Report: {args.output}")
    print(
        f"Coverage: {coverage['samples_recorded']}/{coverage['samples_expected']} samples; "
        f"{coverage['synthesis_failures']} synthesis failure(s)"
    )
    return 0 if coverage["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
