from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

from .catalog import resolve_model
from .errors import ArtifactIntegrityError, OfflineModelError
from .types import InstalledModel

_CHUNK = 1024 * 1024


def _default_cache_dir() -> Path:
    return Path(user_cache_dir("inflectsynth"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, artifact: dict[str, Any]) -> None:
    expected_size = int(artifact["size"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ArtifactIntegrityError(
            f"{path.name}: expected {expected_size} bytes, got {actual_size}"
        )
    actual_sha = _sha256(path)
    if actual_sha != artifact["sha256"]:
        raise ArtifactIntegrityError(
            f"{path.name}: expected SHA-256 {artifact['sha256']}, got {actual_sha}"
        )


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "inflectsynth/0.1"})
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with urllib.request.urlopen(request) as response, temp_path.open("wb") as out:
            shutil.copyfileobj(response, out, length=_CHUNK)
        temp_path.replace(destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def install_model(
    model: str = "nano-v2",
    *,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
    offline: bool = False,
    force_download: bool = False,
) -> InstalledModel:
    record = resolve_model(model, catalog_path=catalog_path)
    root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    target = root / "models" / record["id"] / record["upstream"]["revision"]
    target.mkdir(parents=True, exist_ok=True)

    role_paths: dict[str, Path] = {}
    for artifact in record["artifacts"]:
        path = target / artifact["filename"]
        role_paths[artifact["role"]] = path
        valid = False
        if path.is_file() and not force_download:
            try:
                _verify(path, artifact)
                valid = True
            except ArtifactIntegrityError:
                if offline:
                    raise
        if not valid:
            if offline:
                raise OfflineModelError(f"Missing verified offline artifact: {path}")
            _download(artifact["url"], path)
            _verify(path, artifact)

    return InstalledModel(
        model_id=record["id"],
        root=target,
        duration_path=role_paths["duration"],
        decode_path=role_paths["decode"],
        metadata=record,
    )
