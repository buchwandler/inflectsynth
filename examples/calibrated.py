from inflectsynth import DEFAULT_MODEL, InflectVoice, SynthesisConfig, VoiceLevelConfig

config = SynthesisConfig(voice_level=VoiceLevelConfig(mode="calibrated"))

with InflectVoice.from_pretrained(DEFAULT_MODEL, providers="cpu") as model:
    result = model.synthesize("Hello from calibrated InflectSynth.", config=config)
    result.save_wav("inflect-nano-calibrated.wav")
    print(result.metadata["voice_level"])
