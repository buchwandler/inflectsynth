from inflectsynth import InflectVoice

with InflectVoice.from_pretrained("nano-v2", providers="cpu") as model:
    result = model.synthesize("Hello from InflectSynth.", seed=7)
    result.save_wav("inflect-nano.wav")
