from __future__ import annotations

from pathlib import Path
from typing import Any

from ._onnxvoice import catalog_items, catalog_voices
from .types import ModelInfo, VoiceInfo


def list_models(
    *,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
    catalog_url: str | None = None,
) -> tuple[ModelInfo, ...]:
    """Compatibility discovery adapter backed exclusively by ONNXVoice."""
    items = catalog_items(
        language=language,
        offline=offline,
        refresh=refresh,
        cache_dir=cache_dir,
        catalog_url=catalog_url,
    )
    voices = catalog_voices(
        language=language,
        offline=offline,
        refresh=refresh,
        cache_dir=cache_dir,
        catalog_url=catalog_url,
    )
    voices_by_model: dict[str, list[Any]] = {}
    for record in voices:
        voices_by_model.setdefault(record.asset_id, []).append(record)

    models: list[ModelInfo] = []
    for item in items:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        upstream = metadata.get("upstream")
        upstream = upstream if isinstance(upstream, dict) else {}
        voice_details = metadata.get("voice_details")
        voice_details = voice_details if isinstance(voice_details, dict) else {}
        model_voices = tuple(
            VoiceInfo(
                id=record.voice_id,
                name=str(voice_details.get(record.voice_id, {}).get("name", record.voice_id)),
                language=record.metadata.locale,
                gender=record.gender,
                synthetic=bool(voice_details.get(record.voice_id, {}).get("synthetic", True)),
            )
            for record in voices_by_model.get(item.id, [])
        )
        models.append(
            ModelInfo(
                id=item.id,
                name=str(metadata.get("name", item.id)),
                aliases=tuple(item.aliases),
                language=str(metadata.get("language", "en-US")),
                sample_rate=int(item.sample_rate or 24_000),
                default_voice=item.default_voice or "default",
                voices=model_voices,
                revision=str(
                    metadata.get("source_revision")
                    or upstream.get("source_revision")
                    or upstream.get("revision")
                    or ""
                ),
            )
        )
    return tuple(models)
