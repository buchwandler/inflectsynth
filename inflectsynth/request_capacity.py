"""Strict packaged policy for safe managed Inflect request budgets."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

_EXPECTED_MODELS = ("nano-v2", "micro-v2")
_POLICY_CORPUS = "inflectsynth-request-capacity-v1"
_POLICY_FIELDS = {
    "corpus",
    "environment",
    "evidence_sha256",
    "generated_with",
    "models",
    "provenance",
    "revision",
    "runtime_identity",
    "schema",
    "unit",
}
_RUNTIME_IDENTITY_FIELDS = {
    "engine",
    "engine_version",
    "g2p_revision",
    "runtime_revision",
    "request_api_version",
}
_PROVENANCE_FIELDS = {"model_id", "model_ref", "source_revision", "artifacts"}
_MODEL_POLICY_FIELDS = {
    "maximum",
    "basis",
    "corpus",
    "safe_observation_boundary",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant is not permitted: {value}")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_fields(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has an invalid shape")
    return value


def _validate_policy(document: Any) -> RequestCapacityPolicy:
    root = _require_fields(document, _POLICY_FIELDS, "request-capacity policy")
    if type(root["schema"]) is not int or root["schema"] != 1:
        raise ValueError("request-capacity policy schema must be 1")
    if root["unit"] != "model_tokens":
        raise ValueError("request-capacity policy unit must be 'model_tokens'")
    corpus = _require_string(root["corpus"], "policy corpus")
    if corpus != _POLICY_CORPUS:
        raise ValueError(f"policy corpus must be {_POLICY_CORPUS!r}")

    revision = _require_digest(root["revision"], "policy revision")
    expected_revision = _canonical_sha256(
        {key: value for key, value in root.items() if key != "revision"}
    )
    if revision != expected_revision:
        raise ValueError("policy revision does not match its contents")
    evidence_sha256 = _require_digest(root["evidence_sha256"], "evidence_sha256")

    generated_with = _require_fields(
        root["generated_with"], {"inflectsynth", "inflectg2p", "onnxvoice"}, "generated_with"
    )
    for name, value in generated_with.items():
        _require_string(value, f"generated_with.{name}")
        if value == "unknown":
            raise ValueError(f"generated_with.{name} must identify a package version")

    environment = _require_fields(
        root["environment"], {"python_version", "platform"}, "environment"
    )
    for name, value in environment.items():
        _require_string(value, f"environment.{name}")

    runtime_identity = _require_fields(
        root["runtime_identity"], _RUNTIME_IDENTITY_FIELDS, "runtime_identity"
    )
    for name, value in runtime_identity.items():
        _require_string(value, f"runtime_identity.{name}")
    if runtime_identity["engine"] != "inflect":
        raise ValueError("runtime_identity.engine must be 'inflect'")

    provenance = root["provenance"]
    if not isinstance(provenance, list) or len(provenance) != len(_EXPECTED_MODELS):
        raise ValueError("provenance must contain one record for each canonical model")
    provenance_by_model: dict[str, Mapping[str, Any]] = {}
    for index, raw_record in enumerate(provenance):
        record = _require_fields(raw_record, _PROVENANCE_FIELDS, f"provenance[{index}]")
        model_id = record["model_id"]
        if model_id not in _EXPECTED_MODELS or model_id in provenance_by_model:
            raise ValueError(f"provenance has a non-canonical or duplicate model ID: {model_id!r}")
        if record["model_ref"] != f"inflect:{model_id}":
            raise ValueError(f"provenance model_ref is invalid for {model_id!r}")
        _require_string(record["source_revision"], f"provenance[{index}].source_revision")
        artifacts = _require_fields(
            record["artifacts"], {"duration", "decode"}, f"provenance[{index}].artifacts"
        )
        for artifact_name, raw_artifact in artifacts.items():
            artifact = _require_fields(
                raw_artifact, {"bytes", "sha256"}, f"provenance[{index}].artifacts.{artifact_name}"
            )
            if type(artifact["bytes"]) is not int or artifact["bytes"] <= 0:
                raise ValueError(f"{artifact_name} artifact bytes must be a positive integer")
            _require_digest(artifact["sha256"], f"{artifact_name} artifact sha256")
        provenance_by_model[model_id] = record
    if set(provenance_by_model) != set(_EXPECTED_MODELS):
        raise ValueError("provenance must cover exactly nano-v2 and micro-v2")

    raw_models = _require_fields(root["models"], set(_EXPECTED_MODELS), "models")
    capacities: dict[str, RequestCapacity] = {}
    for model_id in _EXPECTED_MODELS:
        raw_model = _require_fields(
            raw_models[model_id], _MODEL_POLICY_FIELDS, f"models.{model_id}"
        )
        maximum = raw_model["maximum"]
        boundary = raw_model["safe_observation_boundary"]
        if type(maximum) is not int or maximum <= 0:
            raise ValueError(f"models.{model_id}.maximum must be a positive integer")
        if type(boundary) is not int or boundary <= 0:
            raise ValueError(
                f"models.{model_id}.safe_observation_boundary must be a positive integer"
            )
        if maximum > boundary:
            raise ValueError(f"models.{model_id}.maximum exceeds its safe observation boundary")
        if raw_model["basis"] != "validated-safe-request-budget":
            raise ValueError(f"models.{model_id}.basis is unsupported")
        if raw_model["corpus"] != corpus:
            raise ValueError(f"models.{model_id}.corpus does not match the policy corpus")
        source_record = provenance_by_model[model_id]
        capacities[model_id] = RequestCapacity(
            model_id=model_id,
            maximum=maximum,
            policy_revision=revision,
            evidence_sha256=evidence_sha256,
            basis=raw_model["basis"],
            safe_observation_boundary=boundary,
            corpus=corpus,
            source_revision=source_record["source_revision"],
            artifacts=source_record["artifacts"],
        )

    return RequestCapacityPolicy(
        schema=1,
        unit="model_tokens",
        corpus=corpus,
        revision=revision,
        evidence_sha256=evidence_sha256,
        generated_with=generated_with,
        environment=environment,
        runtime_identity=runtime_identity,
        provenance=tuple(provenance),
        models=capacities,
    )


@dataclass(frozen=True, slots=True)
class RequestCapacity:
    """Evidence-backed safe request budget for one canonical managed model."""

    model_id: str
    maximum: int
    policy_revision: str
    evidence_sha256: str
    basis: str
    safe_observation_boundary: int
    corpus: str
    source_revision: str
    artifacts: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", _freeze_json(self.artifacts))


@dataclass(frozen=True, slots=True)
class RequestCapacityPolicy:
    """Immutable, fully validated packaged managed-model capacity policy."""

    schema: int
    unit: Literal["model_tokens"]
    corpus: str
    revision: str
    evidence_sha256: str
    generated_with: Mapping[str, Any]
    environment: Mapping[str, Any]
    runtime_identity: Mapping[str, Any]
    provenance: tuple[Mapping[str, Any], ...]
    models: Mapping[str, RequestCapacity]

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_with", _freeze_json(self.generated_with))
        object.__setattr__(self, "environment", _freeze_json(self.environment))
        object.__setattr__(self, "runtime_identity", _freeze_json(self.runtime_identity))
        object.__setattr__(self, "provenance", _freeze_json(self.provenance))
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))

    def for_model(self, model_id: str) -> RequestCapacity | None:
        """Return a capacity only for an exact canonical model ID."""
        return self.models.get(model_id)


def load_request_capacity_policy(source: str | Path | None = None) -> RequestCapacityPolicy:
    """Load and validate a policy file, or the packaged policy when source is omitted."""
    try:
        if source is None:
            text = (
                resources.files("inflectsynth")
                .joinpath("data")
                .joinpath("request_capacity.json")
                .read_text(encoding="utf-8")
            )
        else:
            text = Path(source).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read request-capacity policy: {error}") from error
    try:
        document = json.loads(text, object_pairs_hook=_json_pairs, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid request-capacity policy JSON: {error}") from error
    return _validate_policy(document)


@lru_cache(maxsize=1)
def default_request_capacity_policy() -> RequestCapacityPolicy:
    """Return the cached, validated packaged policy."""
    return load_request_capacity_policy()


def request_capacity_for(model_id: str) -> RequestCapacity | None:
    """Look up a policy entry by exact canonical managed model ID."""
    if not isinstance(model_id, str):
        return None
    return default_request_capacity_policy().for_model(model_id)


__all__ = [
    "RequestCapacity",
    "RequestCapacityPolicy",
    "default_request_capacity_policy",
    "load_request_capacity_policy",
    "request_capacity_for",
]
