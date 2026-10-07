from __future__ import annotations

import pytest

from inflectsynth import VoiceCalibrationKey, default_voice_calibration


def test_packaged_voice_calibration_covers_both_managed_v2_models() -> None:
    catalog = default_voice_calibration()

    assert catalog.schema == 1
    assert catalog.method == "bs1770"
    assert catalog.corpus == "inflectsynth-prepared-speech-v1"
    assert catalog.reference_lufs == -24.0

    expected = {
        VoiceCalibrationKey("inflect", "nano-v2", "default"): (
            -0.042377569773361046,
            -23.95762243022664,
            0.2913390390127759,
        ),
        VoiceCalibrationKey("inflect", "micro-v2", "default"): (
            -0.3353773192815517,
            -23.66462268071845,
            0.28835105130528405,
        ),
    }
    assert set(catalog.voices) == set(expected)

    for key, (gain_db, measured_lufs, mad_lu) in expected.items():
        record = catalog.voices[key]
        assert record.gain_db == pytest.approx(gain_db, abs=1e-12)
        assert record.measured_lufs == pytest.approx(measured_lufs, abs=1e-12)
        assert record.reference_lufs == -24.0
        assert record.mad_lu == pytest.approx(mad_lu, abs=1e-12)
        assert record.samples == 9
        assert record.method == "bs1770"
        assert record.corpus_version == "inflectsynth-prepared-speech-v1"
