import numpy as np
import pytest

from inflectsynth.errors import InvalidVoiceError
from inflectsynth.types import InstalledModel
from inflectsynth.voice import InflectVoice


class FakeG2P:
    class Result:
        normalized_text = "hello"
        phoneme_text = "həlˈoʊ"
        token_ids = (0, 1, 0)

    def phonemize(self, text):
        return self.Result()


class FakeRuntime:
    providers = ("CPUExecutionProvider",)

    def infer_chunk(self, token_ids, *, speed, variation, seed):
        return np.asarray([0.0, 0.25, 0.0], dtype=np.float32)


def _voice(tmp_path):
    installed = InstalledModel(
        model_id="nano-v2",
        root=tmp_path,
        duration_path=tmp_path / "duration.onnx",
        decode_path=tmp_path / "decode.onnx",
        metadata={
            "id": "nano-v2",
            "sample_rate": 24000,
            "default_voice": "default",
            "voices": {
                "default": {
                    "id": "default",
                    "name": "Default",
                    "language": "en-US",
                    "gender": "male",
                    "synthetic": True,
                }
            },
            "upstream": {"revision": "a" * 40},
        },
    )
    return InflectVoice(runtime=FakeRuntime(), installed=installed, g2p=FakeG2P())


def test_fixed_voice_contract(tmp_path):
    voice = _voice(tmp_path)
    assert voice.available_voices == ("default",)
    result = voice.synthesize("hello", seed=5)
    assert result.sample_rate == 24000
    with pytest.raises(InvalidVoiceError):
        voice.synthesize("hello", voice="someone-else")
