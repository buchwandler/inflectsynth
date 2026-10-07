"""Public Inflect model discovery, backed by ONNXVoice's catalog-only APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._onnxvoice import catalog_items, catalog_voices
from .errors import (
    CatalogDiscoveryError,
    CatalogUnavailableError,
    InflectSynthError,
)


@dataclass(frozen=True, slots=True)
class DescribedVoice:
    id: str
    gender: str = "unknown"
    language: str = "en-US"
    locale: str = "en-US"
    language_label: str = "English"
    languages: tuple[str, ...] = ("en-US",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "languages", tuple(self.languages))


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    id: str
    display_name: str
    version: str | None
    language: str
    sample_rate: int
    aliases: tuple[str, ...]
    voices: tuple[DescribedVoice, ...]
    default_voice: str | None
    source_revision: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "voices", tuple(self.voices))
        object.__setattr__(self, "metadata", _freeze_metadata(dict(self.metadata)))


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_metadata(item) for item in value)
    return value


def _string_values(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,) if values else ()
    if not isinstance(values, Sequence):
        return ()
    return tuple(value for value in values if isinstance(value, str) and value)


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _described_voice(record: Any, item_metadata: Mapping[str, Any]) -> DescribedVoice:
    voice_id = str(getattr(record, "voice_id", ""))
    metadata = getattr(record, "metadata", None)
    details = item_metadata.get("voice_details", {})
    detail = details.get(voice_id, {}) if isinstance(details, Mapping) else {}
    language = _value(metadata, "language", None) or detail.get("language") or "en-US"
    locale = _value(metadata, "locale", None) or detail.get("locale") or language
    languages = _string_values(getattr(record, "languages", ()))
    if not languages:
        languages = (locale,)
    return DescribedVoice(
        id=voice_id,
        gender=str(_value(metadata, "gender", None) or detail.get("gender") or "unknown"),
        language=str(locale or language),
        locale=str(locale),
        language_label=str(
            _value(metadata, "language_label", None) or detail.get("language_label") or locale
        ),
        languages=languages,
    )


def _model_from_item(item: Any, records: Sequence[Any]) -> DiscoveredModel:
    raw_metadata = getattr(item, "metadata", None)
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    model_id = getattr(item, "id", None)
    if not isinstance(model_id, str) or not model_id:
        raise CatalogDiscoveryError("ONNXVoice returned an Inflect catalog item without an id")
    item_system = getattr(item, "system", "inflect")
    matching = tuple(
        record
        for record in records
        if getattr(record, "asset_id", None) == model_id
        and getattr(record, "system", item_system) == item_system
    )
    voices_by_id = {
        voice.id: voice for record in matching if (voice := _described_voice(record, metadata)).id
    }
    voice_details = metadata.get("voice_details")
    detail_ids = tuple(voice_details) if isinstance(voice_details, Mapping) else ()
    declared_ids = _string_values(getattr(item, "voices", ()))
    voice_ids = tuple(dict.fromkeys((*declared_ids, *detail_ids, *voices_by_id)))
    voices = tuple(
        voices_by_id.get(
            voice_id,
            DescribedVoice(
                id=voice_id,
                gender=str(
                    voice_details.get(voice_id, {}).get("gender", "unknown")
                    if isinstance(voice_details, Mapping)
                    else "unknown"
                ),
            ),
        )
        for voice_id in voice_ids
    )

    raw_sample_rate = getattr(item, "sample_rate", None) or metadata.get("sample_rate")
    if (
        isinstance(raw_sample_rate, bool)
        or not isinstance(raw_sample_rate, int)
        or raw_sample_rate <= 0
    ):
        raw_sample_rate = 24_000
    upstream = metadata.get("upstream")
    upstream = upstream if isinstance(upstream, Mapping) else {}
    source_revision = metadata.get("source_revision") or upstream.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision:
        source_revision = None
    version = metadata.get("version")
    if not isinstance(version, str):
        version = None
    aliases = _string_values(getattr(item, "aliases", ()))
    language = metadata.get("language")
    if not isinstance(language, str) or not language:
        language = "en-US"
    default_voice = getattr(item, "default_voice", None) or metadata.get("default_voice")
    if not isinstance(default_voice, str) or not default_voice:
        default_voice = voices[0].id if voices else None
    if source_revision is not None:
        metadata["source_revision"] = source_revision

    return DiscoveredModel(
        id=model_id,
        display_name=str(metadata.get("name") or model_id),
        version=version,
        language=language,
        sample_rate=raw_sample_rate,
        aliases=aliases,
        voices=voices,
        default_voice=default_voice,
        source_revision=source_revision,
        metadata=metadata,
    )


def discover_models(
    *,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
    catalog_url: str | None = None,
) -> tuple[DiscoveredModel, ...]:
    """List catalog models and voices without installing or opening model assets."""
    try:
        items = catalog_items(
            language=language,
            offline=offline,
            refresh=refresh,
            cache_dir=cache_dir,
            catalog_url=catalog_url,
        )
        records = catalog_voices(
            language=language,
            offline=offline,
            refresh=refresh,
            cache_dir=cache_dir,
            catalog_url=catalog_url,
        )
        return tuple(
            _model_from_item(item, records)
            for item in items
            if getattr(item, "system", "inflect") == "inflect"
        )
    except InflectSynthError:
        raise
    except Exception as exc:
        raise CatalogUnavailableError(f"Inflect model catalog is unavailable: {exc}") from exc


__all__ = ["DescribedVoice", "DiscoveredModel", "discover_models"]
