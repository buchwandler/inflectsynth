import sys
from types import SimpleNamespace

import numpy as np

from inflectsynth.runtime import InflectOnnxRuntime


class FakeSession:
    created = []

    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.calls = []
        self.__class__.created.append(self)

    def run(self, outputs, feeds):
        self.calls.append((outputs, feeds))
        if outputs == ["m_p_exp", "logs_p_exp", "y_mask"]:
            shape = (1, 2, 3)
            return [
                np.ones(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
                np.ones(shape, dtype=np.float32),
            ]
        assert outputs == ["waveform"]
        return [np.asarray([[[0.0, 0.25, -0.25]]], dtype=np.float32)]


def test_split_onnx_abi_and_seeded_noise(monkeypatch, tmp_path):
    FakeSession.created.clear()
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    runtime = InflectOnnxRuntime(
        tmp_path / "duration.onnx",
        tmp_path / "decode.onnx",
        providers="cpu",
    )
    audio_a = runtime.infer_chunk((0, 4, 0), speed=2.0, variation=0.5, seed=7)
    decode_a = FakeSession.created[1].calls[-1][1]
    noise_a = decode_a["zp_noise"].copy()
    audio_b = runtime.infer_chunk((0, 4, 0), speed=2.0, variation=0.5, seed=7)
    decode_b = FakeSession.created[1].calls[-1][1]

    duration_outputs, duration_inputs = FakeSession.created[0].calls[0]
    assert duration_outputs == ["m_p_exp", "logs_p_exp", "y_mask"]
    assert duration_inputs["tokens"].dtype == np.int64
    assert duration_inputs["tokens"].shape == (1, 3)
    assert duration_inputs["lengths"].tolist() == [3]
    assert float(duration_inputs["length_scale"]) == 0.5
    assert decode_a["noise_scale"].dtype == np.float32
    assert np.array_equal(noise_a, decode_b["zp_noise"])
    assert np.array_equal(audio_a, audio_b)
