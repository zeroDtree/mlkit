# Docstring style

Public API docstrings are rendered by [mkdocstrings](api/index.md). Types come from **annotations** (`annotations_path: brief` in `mkdocs.yml`), not from repeated `(int, optional)` lines in `Args`.

## Summary line

One sentence describing what the callable does. Use imperative or “Return …” consistently within a module.

## Args

```python
Args:
    checkpoint_dir: Latest checkpoint directory, or ``None`` when there is none.
```

- **No** parameter types in the docstring.
- Defaults are optional when obvious from the signature.

## Returns

Use a short prose paragraph when the return value needs explanation. Omit the entire `Returns:` section when the annotation is `-> None` and adds nothing.

## Raises

```python
Raises:
    ValueError: When ``reporter`` is not a known backend name.
```

## Cross-references

Sphinx-style markup works in API pages: ``:func:`run_experiment` ``, `` ``mlflow`` ``.

## Long constructors

Classes such as `TrainingConfig` document behavior in [Training loop](training.md) instead of listing every field in `__init__`.

## IDE template

Project root [`.mustache`](../.mustache) is wired in [`.vscode/settings.json`](../.vscode/settings.json) for the autoDocstring extension.

To bulk-fix legacy ``Args`` type lines in existing docstrings:

```bash
python3 scripts/normalize_docstrings.py
```

Canonical examples: `mlkit.util.report.run_file`, `mlkit.util.failure`, `mlkit.util.experiment.types`.

Example entry scripts (`example/**/main*.py`) place the **module docstring first**, then `sys.path` setup, then remaining imports (Ruff E402).