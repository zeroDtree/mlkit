# mlkit

`mlkit` is a machine learning toolkit providing components for diffusion models, flow matching, pipelines, datasets, and more.

## Quick Start

```python
import mlkit
```

Hydra examples use `run_experiment` with a `TrainSpec` (training) or an `InferSpec` (inference); see the guides below before diving into the API.

## Guides

Recommended reading order:

1. [Running an experiment](experiment.md) — Hydra config, `run_experiment`, `TrainSpec`, `InferSpec`, custom tasks, artifacts paths
2. [Experiment tracking](reporter.md) — MLflow, WandB, resume via `report_run.json`
3. [Training loop](training.md) — epochs vs steps, checkpoints, callbacks, halt
4. [Learning rate](learning-rate.md) — scheduler length under Accelerate

Contributors: [Docstring style](docstrings.md) for API docstrings and the autoDocstring template.

## API Reference

- [API Reference](api/index.md) — Auto-generated module documentation (`experiment` and `report` have dedicated pages)