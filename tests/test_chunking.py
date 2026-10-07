import numpy as np

from inflectsynth.voice import boundary_pause_seconds, edge_fade, split_text


def test_upstream_chunking_helpers():
    assert split_text("One. Two?") == ["One.", "Two?"]
    assert boundary_pause_seconds("Question?") == 0.28
    audio = np.ones(240, dtype=np.float32)
    faded = edge_fade(audio, 24000, 5.0)
    assert faded[0] == 0.0
    assert faded[-1] == 0.0
