from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .errors import CatalogError, ModelNotFoundError
from .types import ModelInfo, VoiceInfo


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        text = files("inflectsynth").joinpath("data/models.json").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogError("Invalid Inflect catalog JSON") from exc
    if data.get("schema") != 1 or data.get("kind") != "inflect-onnx-model-catalog":
        raise CatalogError("Unsupported Inflect catalog")
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise CatalogError("Inflect catalog has no models")
    return data


def resolve_model(model: str, *, catalog_path: str | Path | None = None) -> dict[str, Any]:
    data = load_catalog(catalog_path)
    requested = model.strip()
    for model_id, record in data["models"].items():
        if requested == model_id or requested in record.get("aliases", []):
            return record
    raise ModelNotFoundError(f"Unknown Inflect model: {model!r}")


def list_models(*, catalog_path: str | Path | None = None) -> tuple[ModelInfo, ...]:
    data = load_catalog(catalog_path)
    result: list[ModelInfo] = []
    for model_id, record in data["models"].items():
        voices = tuple(
            VoiceInfo(
                id=voice["id"],
                name=voice["name"],
                language=voice["language"],
                gender=voice["gender"],
                synthetic=bool(voice["synthetic"]),
            )
            for voice in record["voices"].values()
        )
        result.append(
            ModelInfo(
                id=model_id,
                name=record["name"],
                aliases=tuple(record.get("aliases", [])),
                language=record["language"],
                sample_rate=int(record["sample_rate"]),
                default_voice=record["default_voice"],
                voices=voices,
                revision=record["upstream"]["revision"],
            )
        )
    return tuple(result)
