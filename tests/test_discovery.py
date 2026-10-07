from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import inflectsynth.discovery as discovery
from inflectsynth import CatalogDiscoveryError, CatalogUnavailableError, discover_models


def _records() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    items = tuple(
        SimpleNamespace(
            system="inflect",
            id=model_id,
            aliases=(alias,),
            sample_rate=24_000,
            voices=("default",),
            default_voice="default",
            metadata={
                "name": name,
                "version": "2",
                "language": "en-US",
                "source_revision": f"source-{model_id}",
                "controls": {"variation": {"default": 0.667}},
                "runtime": {"profile": "inflect-v2-split-v1"},
                "voice_details": {
                    "default": {
                        "id": "default",
                        "name": "Default",
                        "language": "en",
                        "locale": "en-US",
                        "gender": "female",
                        "synthetic": True,
                    }
                },
            },
        )
        for model_id, alias, name in (
            ("nano-v2", "nano", "Inflect Nano v2"),
            ("micro-v2", "micro", "Inflect Micro v2"),
        )
    )
    voices = tuple(
        SimpleNamespace(
            system="inflect",
            asset_id=item.id,
            voice_id="default",
            metadata=SimpleNamespace(
                language="en",
                locale="en-US",
                language_label="English",
                gender="female",
            ),
            languages=("en-US", "en"),
        )
        for item in items
    )
    return items, voices


def test_discovery_maps_models_voices_and_controls_without_installing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items, voices = _records()
    calls: list[tuple[str, dict[str, Any]]] = []

    def list_items(**kwargs: Any) -> tuple[Any, ...]:
        calls.append(("items", kwargs))
        return items

    def list_voices(**kwargs: Any) -> tuple[Any, ...]:
        calls.append(("voices", kwargs))
        return voices

    monkeypatch.setattr(discovery, "catalog_items", list_items)
    monkeypatch.setattr(discovery, "catalog_voices", list_voices)

    models = discover_models(
        language="en-US",
        offline=True,
        refresh=True,
        catalog_url="catalog.json",
    )

    assert [model.id for model in models] == ["nano-v2", "micro-v2"]
    assert models[0].aliases == ("nano",)
    assert models[0].sample_rate == 24_000
    assert models[0].default_voice == "default"
    assert models[0].source_revision == "source-nano-v2"
    assert models[0].metadata["controls"]["variation"]["default"] == 0.667
    assert models[0].voices == (
        discovery.DescribedVoice(
            id="default",
            gender="female",
            language="en-US",
            locale="en-US",
            language_label="English",
            languages=("en-US", "en"),
        ),
    )
    expected = {
        "language": "en-US",
        "offline": True,
        "refresh": True,
        "cache_dir": None,
        "catalog_url": "catalog.json",
    }
    assert calls == [("items", expected), ("voices", expected)]
    with pytest.raises(TypeError):
        models[0].metadata["changed"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        models[0].metadata["controls"]["variation"]["default"] = 0.5  # type: ignore[index]


def test_catalog_errors_are_translated_to_public_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        discovery,
        "catalog_items",
        lambda **_kwargs: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    with pytest.raises(CatalogUnavailableError):
        discover_models(offline=True)

    error = CatalogDiscoveryError("invalid catalog")
    monkeypatch.setattr(discovery, "catalog_items", lambda **_kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(CatalogDiscoveryError) as caught:
        discover_models()
    assert caught.value is error
