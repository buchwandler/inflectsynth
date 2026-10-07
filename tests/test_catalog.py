from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import inflectsynth.catalog as catalog


def test_compatibility_discovery_adapts_onnxvoice_catalog_records(monkeypatch: Any) -> None:
    item = SimpleNamespace(
        id="nano-v2",
        aliases=("nano",),
        sample_rate=24_000,
        default_voice="default",
        metadata={
            "name": "Inflect Nano v2",
            "language": "en-US",
            "source_revision": "source-revision",
            "voice_details": {
                "default": {"name": "Default", "synthetic": True},
            },
        },
    )
    voice = SimpleNamespace(
        asset_id="nano-v2",
        voice_id="default",
        metadata=SimpleNamespace(locale="en-US"),
        gender="female",
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    def get_items(**kwargs: Any) -> tuple[object, ...]:
        calls.append(("items", kwargs))
        return (item,)

    def get_voices(**kwargs: Any) -> tuple[object, ...]:
        calls.append(("voices", kwargs))
        return (voice,)

    monkeypatch.setattr(catalog, "catalog_items", get_items)
    monkeypatch.setattr(catalog, "catalog_voices", get_voices)

    models = catalog.list_models(language="en-US", offline=True, refresh=True)

    assert len(models) == 1
    model = models[0]
    assert (model.id, model.name, model.aliases) == ("nano-v2", "Inflect Nano v2", ("nano",))
    assert (model.language, model.sample_rate, model.default_voice) == ("en-US", 24_000, "default")
    assert model.revision == "source-revision"
    assert model.voices[0].id == "default"
    assert model.voices[0].language == "en-US"
    assert model.voices[0].gender == "female"
    options = {
        "language": "en-US",
        "offline": True,
        "refresh": True,
        "cache_dir": None,
        "catalog_url": None,
    }
    assert calls == [("items", options), ("voices", options)]
