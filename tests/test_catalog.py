from inflectsynth import list_models, resolve_model


def test_bootstrap_catalog_has_both_v2_models_and_complete_fixed_voice_inventory():
    models = {model.id: model for model in list_models()}
    assert set(models) == {"nano-v2", "micro-v2"}
    assert models["nano-v2"].default_voice == "default"
    assert tuple(voice.id for voice in models["nano-v2"].voices) == ("default",)
    assert tuple(voice.id for voice in models["micro-v2"].voices) == ("default",)
    assert resolve_model("nano")["id"] == "nano-v2"
    assert resolve_model("micro")["id"] == "micro-v2"
