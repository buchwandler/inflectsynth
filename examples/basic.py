from inflectsynth import DEFAULT_MODEL, InflectVoice

with InflectVoice.from_pretrained(DEFAULT_MODEL, providers="cpu") as model:
    result = model.synthesize("Hello from InflectSynth.", seed=7)
    result.save_wav("inflect-nano.wav")
