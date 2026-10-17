from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from inflectsynth import DEFAULT_MODEL, InflectVoice
from inflectsynth._onnxvoice import DEFAULT_CATALOG_URL
from inflectsynth.errors import OnnxVoiceContractError

ROOT = Path(__file__).resolve().parents[1]
CATALOG_FIXTURE = ROOT / "tests" / "fixtures" / "inflect_nano_catalog.json"
MODEL_REVISION = "df3627e44c26192714c9dcba6b8e9b47e7a6e3d9"
SOURCE_REVISION = "67fc9249e0bb8ab4b196d25cb09b6dd43f5e4f13"
EXPECTED_SHA256 = {
    "duration": "6e5fe09584f979dcd2da4f89d5274385ad5197efcd5ddcaa5ac03db637d7451e",
    "decode": "f98b1ba21f3d5306ee6cf96a02ba9783082ef99314e77f1ee0c03954dc7de2b8",
}


def _local_catalog(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    catalog = json.loads(CATALOG_FIXTURE.read_text(encoding="utf-8"))
    model: dict[str, Any] = catalog["models"]["nano-v2"]
    assert model["upstream"]["revision"] == MODEL_REVISION
    assert model["upstream"]["source_revision"] == SOURCE_REVISION
    assert {
        artifact["role"]: artifact["sha256"] for artifact in model["artifacts"]
    } == EXPECTED_SHA256

    payloads: dict[str, bytes] = {}
    for artifact in model["artifacts"]:
        payload = f"network-free fixture for {artifact['role']}".encode()
        payloads[artifact["filename"]] = payload
        artifact["size"] = len(payload)
        artifact["sha256"] = hashlib.sha256(payload).hexdigest()

    catalog_path = tmp_path / "models.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog_path, payloads


def test_from_pretrained_uses_managed_inflect_assets_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path, payloads = _local_catalog(tmp_path)
    download_calls: list[tuple[str, str, str]] = []

    def download(source: Any, *, local_dir: Path, offline: bool) -> Path:
        assert not offline
        download_calls.append((source.repository, source.revision, source.path))
        target = local_dir / Path(source.path).name
        target.write_bytes(payloads[target.name])
        return target

    monkeypatch.setattr("onnxvoice.store.download_huggingface_file", download)
    voice = InflectVoice.from_pretrained(
        "nano-v2",
        cache_dir=tmp_path / "cache",
        catalog_url=str(catalog_path),
        providers="cpu",
    )
    try:
        installation = voice.installed.installation
        assert DEFAULT_MODEL == "nano-v2"
        assert DEFAULT_CATALOG_URL == (
            "https://raw.githubusercontent.com/buchwandler/inflect-onnx-bundles/"
            "main/catalog/models.json"
        )
        assert voice.model_ref == "inflect:nano-v2"
        assert installation.system == "inflect"
        assert voice.metadata["source_revision"] == SOURCE_REVISION
        assert {artifact.role for artifact in installation.artifacts} == {"duration", "decode"}
        assert download_calls == [
            ("owensong/Inflect-Nano-v2-ONNX", MODEL_REVISION, "onnx/duration.onnx"),
            ("owensong/Inflect-Nano-v2-ONNX", MODEL_REVISION, "onnx/decode.onnx"),
        ]
        for role in ("duration", "decode"):
            artifact = installation.require_artifact(role)
            expected = payloads[f"{role}.onnx"]
            assert artifact.path.read_bytes() == expected
            assert artifact.sha256 == hashlib.sha256(expected).hexdigest()
    finally:
        voice.close()


def test_missing_huggingface_dependency_preserves_actionable_error_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path, _ = _local_catalog(tmp_path)
    import onnxvoice.huggingface as onnxvoice_huggingface
    from onnxvoice.errors import OptionalDependencyError

    real_import_module = importlib.import_module

    def import_module(name: str, package: str | None = None) -> Any:
        if name == "huggingface_hub":
            raise ModuleNotFoundError("No module named 'huggingface_hub'", name=name)
        return real_import_module(name, package)

    monkeypatch.setattr(onnxvoice_huggingface.importlib, "import_module", import_module)
    with pytest.raises(OnnxVoiceContractError) as caught:
        InflectVoice.from_pretrained(
            "nano-v2",
            cache_dir=tmp_path / "cache",
            catalog_url=str(catalog_path),
        )

    error = caught.value
    assert "huggingface_hub" in str(error)
    assert "onnxvoice[huggingface]" in str(error)
    assert "unsupported model" not in str(error).lower()
    assert "capacity" not in str(error).lower()
    assert isinstance(error.__cause__, OptionalDependencyError)
    assert isinstance(error.__cause__.__cause__, ModuleNotFoundError)
