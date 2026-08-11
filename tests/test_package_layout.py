from pathlib import Path

import dynnav


def test_dynnav_import_resolves_to_canonical_package() -> None:
    repository = Path(__file__).resolve().parents[1]
    package_file = Path(dynnav.__file__).resolve()
    assert package_file.is_relative_to(repository / "dynnav")
    assert dynnav.__version__ == "0.2.0"


def test_legacy_source_tree_contains_no_python_package() -> None:
    repository = Path(__file__).resolve().parents[1]
    legacy = repository / "src" / "dynnav"
    assert not list(legacy.rglob("*.py"))
