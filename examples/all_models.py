from inflectsynth import InflectVoice, list_models

for info in list_models():
    with InflectVoice.from_pretrained(info.id, providers="cpu") as model:
        result = model.synthesize(f"This is {info.name}.", voice="default", seed=7)
        result.save_wav(f"{info.id}.wav")
