# mlkit

`mlkit` is a machine learning toolkit providing components for diffusion models, flow matching, pipelines, datasets, and more.

## Quick Start

```python
import mlkit
```

Hydra examples use `run_experiment` with an `ExperimentSpec`; see the guides below before diving into the API.

## Guides

Recommended reading order:

1. [Running an experiment](experiment.md) — Hydra config, `ExperimentSpec`, `run_experiment`, artifacts paths
2. [Experiment tracking](reporter.md) — MLflow, WandB, resume via `report_run.json`
3. [Training loop](training.md) — epochs vs steps, checkpoints, callbacks, halt
4. [Learning rate](learning-rate.md) — scheduler length under Accelerate

## API Reference

- [API Reference](api/index.md) — Auto-generated module documentation (`experiment` and `report` have dedicated pages)