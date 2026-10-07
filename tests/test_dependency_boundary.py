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
    assert '"onnxvoice>=0.2.5,<0.3"' in project
    assert '"audiosig>=0.1.4,<0.2"' in project
    assert '"inflectg2p>=0.1.0,<0.2"' in project
    assert 'cpu = ["onnxvoice[cpu]>=0.2.5,<0.3"]' in project
    assert 'gpu = ["onnxvoice[gpu]>=0.2.5,<0.3"]' in project
    assert 'directml = ["onnxvoice[directml]>=0.2.5,<0.3"]' in project
    assert "platformdirs" not in project
    assert '"onnxruntime>=' not in project
