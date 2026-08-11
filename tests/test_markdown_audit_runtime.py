from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import markdown_audit_core  # noqa: E402
from markdown_audit_runtime import install_document_discovery_filter  # noqa: E402


def test_generated_and_transient_documents_are_not_discovered(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    (docs / "DOCUMENTATION_MAP.md").write_text(
        "# Generated map\n\n[stale](../.pytest_cache/README.md)\n",
        encoding="utf-8",
    )
    (docs / "MARKDOWN_INVENTORY.md").write_text("# Generated inventory\n", encoding="utf-8")
    transient = tmp_path / "pip-metadata-example"
    transient.mkdir()
    (transient / "METADATA").write_text("# Package\n\n[missing](AUTHORS.md)\n", encoding="utf-8")
    pytest_output = tmp_path / "pytest-of-root" / "pytest-7" / "case"
    pytest_output.mkdir(parents=True)
    (pytest_output / "README.md").write_text(
        "# Generated test report\n",
        encoding="utf-8",
    )
    install_document_discovery_filter()
    discovered = {
        path.relative_to(tmp_path).as_posix()
        for path in markdown_audit_core.discover_documents(tmp_path)
    }
    assert discovered == {"README.md"}
