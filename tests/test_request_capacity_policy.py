from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import MutableMapping
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any, cast

import pytest

from inflectsynth.request_capacity import (
    RequestCapacity,
    RequestCapacityPolicy,
    default_request_capacity_policy,
    load_request_capacity_policy,
    request_capacity_for,
)


def _packaged_document() -> dict[str, Any]:
    text = (
        resources.files("inflectsynth")
        .joinpath("data")
        .joinpath("request_capacity.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(text)


def _write_document(tmp_path: Path, document: dict[str, Any]) -> Path:
    result = deepcopy(document)
    revision_payload = {key: value for key, value in result.items() if key != "revision"}
    encoded = json.dumps(revision_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["revision"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    path = tmp_path / "request_capacity.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def test_packaged_policy_loads_and_resolves_only_exact_canonical_ids() -> None:
    policy = default_request_capacity_policy()

    assert isinstance(policy, RequestCapacityPolicy)
    assert set(policy.models) == {"nano-v2", "micro-v2"}
    assert policy.unit == "model_tokens"
    assert policy.revision
    assert policy.for_model("nano-v2") == request_capacity_for("nano-v2")
    assert request_capacity_for("nano") is None
    assert request_capacity_for("other-v2") is None
    assert request_capacity_for("Nano-v2") is None

    for model_id in ("nano-v2", "micro-v2"):
        capacity = request_capacity_for(model_id)
        assert isinstance(capacity, RequestCapacity)
        assert capacity.model_id == model_id
        assert type(capacity.maximum) is int and capacity.maximum > 0
        assert capacity.maximum <= capacity.safe_observation_boundary
        assert capacity.source_revision
        assert capacity.evidence_sha256 == policy.evidence_sha256


def test_returned_policy_and_nested_provenance_are_immutable() -> None:
    policy = load_request_capacity_policy()
    capacity = policy.models["nano-v2"]

    with pytest.raises(TypeError):
        operator.setitem(
            cast(MutableMapping[str, RequestCapacity], policy.models), "nano-v2", capacity
        )
    with pytest.raises(TypeError):
        operator.setitem(cast(MutableMapping[str, Any], capacity.artifacts["duration"]), "bytes", 1)
    with pytest.raises(TypeError):
        operator.setitem(
            cast(MutableMapping[str, Any], policy.provenance[0]), "source_revision", "changed"
        )


def test_policy_rejects_malformed_schema_and_unknown_model_ids(tmp_path: Path) -> None:
    document = _packaged_document()
    document["schema"] = True
    with pytest.raises(ValueError, match="schema must be 1"):
        load_request_capacity_policy(_write_document(tmp_path, document))

    document = _packaged_document()
    document["models"]["unmanaged"] = document["models"].pop("nano-v2")
    with pytest.raises(ValueError, match="models has an invalid shape"):
        load_request_capacity_policy(_write_document(tmp_path, document))


def test_policy_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":1,"schema":1}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_request_capacity_policy(path)


@pytest.mark.parametrize("maximum", [0, -1, True])
def test_policy_rejects_non_positive_or_boolean_maximum(
    tmp_path: Path, maximum: int | bool
) -> None:
    document = _packaged_document()
    document["models"]["nano-v2"]["maximum"] = maximum

    with pytest.raises(ValueError, match="maximum must be a positive integer"):
        load_request_capacity_policy(_write_document(tmp_path, document))


def test_policy_rejects_empty_invalid_or_stale_revision(tmp_path: Path) -> None:
    document = _packaged_document()
    document["revision"] = ""
    path = tmp_path / "empty-revision.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="policy revision must be a lowercase SHA-256 digest"):
        load_request_capacity_policy(path)

    document = _packaged_document()
    document["models"]["nano-v2"]["maximum"] -= 1
    path = tmp_path / "stale-revision.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="revision does not match its contents"):
        load_request_capacity_policy(path)


def test_policy_rejects_malformed_provenance_and_observation_boundary(tmp_path: Path) -> None:
    document = _packaged_document()
    document["provenance"][0]["source_revision"] = ""
    with pytest.raises(ValueError, match="source_revision must be a non-empty string"):
        load_request_capacity_policy(_write_document(tmp_path, document))

    document = _packaged_document()
    document["models"]["nano-v2"]["safe_observation_boundary"] = True
    with pytest.raises(ValueError, match="safe_observation_boundary must be a positive integer"):
        load_request_capacity_policy(_write_document(tmp_path, document))

    document = _packaged_document()
    document["models"]["nano-v2"]["maximum"] = (
        document["models"]["nano-v2"]["safe_observation_boundary"] + 1
    )
    with pytest.raises(ValueError, match="exceeds its safe observation boundary"):
        load_request_capacity_policy(_write_document(tmp_path, document))
