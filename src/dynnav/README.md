# Legacy Python package location

The installable DynNav package was consolidated into [`../../dynnav`](../../dynnav)
to remove the former dual-package import ambiguity. This directory intentionally
contains no Python source.

Use:

```bash
python -m pip install -e ".[dev]"
python -c "import dynnav; print(dynnav.__file__)"
```

The printed path must resolve under the repository-root `dynnav/` directory.
