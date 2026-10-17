from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "inflectsynth"
FORBIDDEN_IMPORTS = {"onnxruntime", "platformdirs", "urllib"}


def test_production_imports_leave_runtime_assets_and_downloads_to_onnxvoice() -> None:
    violations: list[str] = []
    for source_path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            for name in imported:
                root = name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORTS:
                    violations.append(f"{source_path.relative_to(ROOT)} imports {name}")
        text = source_path.read_text(encoding="utf-8")
        if "urlopen(" in text or "download_huggingface_file" in text:
            violations.append(f"{source_path.relative_to(ROOT)} contains an asset downloader")

    assert not violations, "\n".join(violations)


def test_root_import_does_not_import_onnxvoice_or_onnxruntime() -> None:
    code = (
        "import sys, inflectsynth; "
        "assert 'onnxvoice' not in sys.modules; "
        "assert 'onnxruntime' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_runtime_dependencies_and_provider_extras_delegate_to_onnxvoice() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"onnxvoice[huggingface]>=0.2.10,<0.3"' in project
    assert '"audiosig>=0.1.4,<0.2"' in project
    assert '"inflectg2p>=0.1.0,<0.2"' in project
    assert 'cpu = ["onnxvoice[cpu]>=0.2.10,<0.3"]' in project
    assert 'gpu = ["onnxvoice[gpu]>=0.2.10,<0.3"]' in project
    assert 'directml = ["onnxvoice[directml]>=0.2.10,<0.3"]' in project
    assert "platformdirs" not in project
    assert '"onnxruntime>=' not in project


def test_built_wheel_metadata_installs_huggingface_transport(tmp_path) -> None:
    import subprocess
    import sys
    import zipfile

    from packaging.requirements import Requirement

    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    [wheel_path] = tmp_path.glob("inflectsynth-*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_path = next(
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = wheel.read(metadata_path).decode("utf-8")

    requirements = [
        Requirement(line.removeprefix("Requires-Dist: "))
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: ")
    ]
    base = next(
        requirement
        for requirement in requirements
        if requirement.name.lower() == "onnxvoice" and requirement.marker is None
    )
    assert set(base.extras) == {"huggingface"}
    assert base.specifier.contains("0.2.10")
    assert not base.specifier.contains("0.3")

    for backend in ("cpu", "gpu", "directml"):
        backend_requirement = next(
            requirement
            for requirement in requirements
            if requirement.name.lower() == "onnxvoice"
            and set(requirement.extras) == {backend}
            and requirement.marker is not None
            and requirement.marker.evaluate({"extra": backend})
        )
        assert backend_requirement.specifier.contains("0.2.10")
        assert not backend_requirement.specifier.contains("0.3")
