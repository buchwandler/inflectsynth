from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from inflectsynth import (
    CalibrationDataError,
    SynthesisConfig,
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelCalibration,
    VoiceLevelConfig,
    apply_voice_level_calibration,
    load_voice_calibration,
)
from inflectsynth.errors import InvalidSynthesisConfigError


def _catalog_data(voices: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "schema": 1,
        "method": "bs1770",
        "corpus": "inflectsynth-prepared-speech-v1",
        "reference_lufs": -24.0,
        "generated_with": {"inflectsynth": "0.1.0"},
        "voices": voices or {},
    }


def _write_catalog(tmp_path: Path, data: dict[str, object], name: str = "catalog.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_voice_level_config_defaults_and_rejects_invalid_values() -> None:
    assert VoiceLevelConfig().mode == "off"
    assert VoiceLevelConfig().gain_db is None
    for mode in ("automatic", "", None, 1):
        with pytest.raises(InvalidSynthesisConfigError):
            VoiceLevelConfig(mode=mode)  # type: ignore[arg-type]
    for gain in (float("nan"), float("inf"), -float("inf"), True):
        with pytest.raises(InvalidSynthesisConfigError):
            VoiceLevelConfig(gain_db=gain)
    with pytest.raises(InvalidSynthesisConfigError):
        SynthesisConfig(voice_level=object()).validated()  # type: ignore[arg-type]


def test_calibration_key_requires_exact_inflect_identity() -> None:
    key = VoiceCalibrationKey.parse("inflect:nano-v2:default")
    assert key == VoiceCalibrationKey("inflect", "nano-v2", "default")
    assert str(key) == "inflect:nano-v2:default"
    for invalid in ("inflect:nano-v2", "inflect:nano:v2:default", "piper:nano-v2:default"):
        with pytest.raises(CalibrationDataError):
            VoiceCalibrationKey.parse(invalid)
    with pytest.raises(CalibrationDataError):
        VoiceCalibrationKey("inflect", "nano:v2", "default")


def test_catalog_load_is_strict_deterministic_and_deeply_immutable(tmp_path: Path) -> None:
    key = "inflect:nano-v2:default"
    data = _catalog_data(
        {
            key: {
                "gain_db": -1.5,
                "measured_lufs": -22.5,
                "reference_lufs": -24.0,
                "mad_lu": 0.1,
                "samples": 9,
                "method": "bs1770",
                "corpus_version": "inflectsynth-prepared-speech-v1",
            }
        }
    )
    path = _write_catalog(tmp_path, data)
    loaded = load_voice_calibration(path)
    repeated = load_voice_calibration(_write_catalog(tmp_path, data, "again.json"))
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected_key = VoiceCalibrationKey.parse(key)

    assert loaded.revision == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert loaded.revision == repeated.revision
    assert loaded.voices[expected_key].gain_db == -1.5
    with pytest.raises(TypeError):
        loaded.voices[expected_key] = VoiceLevelCalibration(0.0)  # type: ignore[index]
    with pytest.raises(TypeError):
        loaded.generated_with["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data.update(extra=True), "unknown field"),
        (lambda data: data.update(schema=2), "unsupported calibration schema"),
        (lambda data: data.update(schema=True), "unsupported calibration schema"),
        (lambda data: data.update(method="peak"), "unsupported calibration method"),
        (lambda data: data.update(corpus=""), "corpus must be a non-empty string"),
        (lambda data: data.update(generated_with={"audiosig": 1}), "string mapping"),
        (lambda data: data.update(reference_lufs=float("nan")), "non-finite"),
    ],
)
def test_catalog_rejects_invalid_top_level_data(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    data = _catalog_data()
    mutate(data)  # type: ignore[operator]
    with pytest.raises(CalibrationDataError, match=message):
        load_voice_calibration(_write_catalog(tmp_path, data))


def test_catalog_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":1,"method":"bs1770","corpus":"x",'
        '"reference_lufs":-24,"generated_with":{},"voices":{},"voices":{}}',
        encoding="utf-8",
    )
    with pytest.raises(CalibrationDataError, match="duplicate JSON object key"):
        load_voice_calibration(duplicate)

    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text(
        '{"schema":1,"method":"bs1770","corpus":"x",'
        '"reference_lufs":NaN,"generated_with":{},"voices":{}}',
        encoding="utf-8",
    )
    with pytest.raises(CalibrationDataError, match="non-finite JSON number"):
        load_voice_calibration(nonfinite)


def test_catalog_rejects_unknown_record_fields_and_invalid_measurements(tmp_path: Path) -> None:
    key = "inflect:nano-v2:default"
    invalid_records = (
        ({"gain_db": 0.5, "unexpected": "field"}, "unknown field"),
        ({"gain_db": float("inf")}, "non-finite"),
        ({"gain_db": 0.5, "samples": 0}, "positive integer"),
        ({"gain_db": 0.5, "samples": True}, "positive integer"),
        ({"gain_db": 0.5, "method": "peak"}, "unsupported method"),
        ({"gain_db": 0.5, "corpus_version": 12}, "corpus_version must be a string"),
    )
    for index, (record, message) in enumerate(invalid_records):
        data = _catalog_data({key: record})
        with pytest.raises(CalibrationDataError, match=message):
            load_voice_calibration(_write_catalog(tmp_path, data, f"bad-{index}.json"))


def test_gain_selection_and_static_application_never_clip() -> None:
    key = VoiceCalibrationKey("inflect", "nano-v2", "default")
    catalog = VoiceCalibrationCatalog(
        schema=1,
        method="bs1770",
        corpus="inflectsynth-prepared-speech-v1",
        reference_lufs=-24.0,
        generated_with={},
        voices={key: VoiceLevelCalibration(gain_db=6.0)},
        revision="a" * 64,
    )
    audio = np.asarray([0.8, -0.75], dtype=np.float32)

    off_audio, off = apply_voice_level_calibration(audio, VoiceLevelConfig(), key, catalog=catalog)
    assert np.array_equal(off_audio, audio)
    assert off.source == "off" and not off.applied

    override_audio, override = apply_voice_level_calibration(
        audio, VoiceLevelConfig(mode="calibrated", gain_db=3.0), key, catalog=catalog
    )
    np.testing.assert_allclose(override_audio, audio * (10.0 ** (3.0 / 20.0)))
    assert override_audio.max() > 1.0
    assert override.source == "override"

    measured_audio, measured = apply_voice_level_calibration(
        audio, VoiceLevelConfig(mode="calibrated"), key, catalog=catalog
    )
    np.testing.assert_allclose(measured_audio, audio * (10.0 ** (6.0 / 20.0)))
    assert measured_audio.max() > 1.0
    assert measured.source == "catalog"
    assert measured.catalog_revision == "a" * 64


def test_missing_identity_and_missing_calibration_are_noops(tmp_path: Path) -> None:
    audio = np.asarray([0.1, -0.2], dtype=np.float32)
    config = VoiceLevelConfig(mode="calibrated")
    missing_identity, missing = apply_voice_level_calibration(audio, config, None)
    assert np.array_equal(missing_identity, audio)
    assert missing.source == "missing_identity"
    assert "stable managed" in missing.reason

    key = VoiceCalibrationKey("inflect", "nano-v2", "default")
    catalog = load_voice_calibration(_write_catalog(tmp_path, _catalog_data()))
    missing_record, absent = apply_voice_level_calibration(audio, config, key, catalog=catalog)
    assert np.array_equal(missing_record, audio)
    assert absent.source == "missing_calibration"
    assert absent.catalog_revision == catalog.revision
    assert math.isfinite(absent.gain_db)
