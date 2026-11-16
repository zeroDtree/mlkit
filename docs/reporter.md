# Experiment tracking

Training metrics flow through a `Reporter` attached to the pipeline as a `ReportCallback`. See the [report API](api/report.md). Examples do **not** call `wandb.log` directly from the pipeline; configuration lives in Hydra under `reporter`, `mlflow`, and `wandb`.

For the Hydra entry and teardown hooks, see [Running an experiment](experiment.md). For log intervals and halt behavior inside the loop, see [Training loop](training.md).

## Choosing a backend

Set `reporter` in config (string, case-insensitive):

| Value | Implementation | Notes |
| --- | --- | --- |
| `none` | `NullReporter` | No tracking run; metrics still go to the file logger when configured |
| `mlflow` | `MLflowReporter` | Default in most examples |
| `wandb` | `WandbReporter` | Uses `cfg.wandb` |

[`get_reporter`](api/report.md) builds the backend. The run is not active until `open()`.

### MLflow fields (`cfg.mlflow`)

| Field | Purpose |
| --- | --- |
| `tracking_uri` | e.g. `sqlite:///artifacts/mlflow.db` for local SQLite |
| `artifact_location` | e.g. `artifacts/mlruns` |
| `experiment` | MLflow experiment name |

### WandB fields (`cfg.wandb`)

| Field | Purpose |
| --- | --- |
| `project` | WandB project |
| `entity` | Team or user |
| `group` | Run grouping |
| `mode` | e.g. `offline` |
| `reinit` | Whether to start a fresh run (default `true` in examples) |

Example defaults (`example/butterfly_edm/config.yaml`):

```yaml
reporter: mlflow

mlflow:
  tracking_uri: sqlite:///artifacts/mlflow.db
  artifact_location: artifacts/mlruns
  experiment: test

wandb:
  reinit: true
  mode: offline
  project: test
  group: default
  entity: superposed-tree
```

## Lifecycle

On the **main process only**, `run_experiment`:

1. Asks the task for `run_name` and resolves the Hydra config.
2. Asks the task for `resume_run_id`. `TrainSpec` reads it from the latest checkpoint directory when continuing a run; `InferSpec` always opens a new run named `<training run>-infer`.
3. Enters `ReporterContext`, which calls `open()` on enter and `close(success=...)` on exit.

After the LR scheduler is built, `TrainSpec` calls `reporter.store_config` with the **post-scheduler** config so the tracked run matches what training actually uses. Task hooks reach the same run through `run.reporter`, which is `None` off the main process.

## Pipeline connection

`reporter.as_callback()` registers a `ReportCallback` on the pipeline:

| Event | Action |
| --- | --- |
| `LOG` | `reporter.log(metrics, step=...)` |
| `PRE_FINALIZE_SAVE` | Writes `report_run.json` into the checkpoint directory about to be published |

Other callback events are ignored by the report callback.

## Resume with `report_run.json`

When a checkpoint is published, the main process may write `report_run.json`:

```json
{"backend": "mlflow", "run_id": "<id>"}
```

On the next launch, `resume_run_id(latest_checkpoint_dir, reporter_kind)` returns that id **only** when the file’s `backend` matches the configured `reporter`. The reporter reopens or attaches to the same run.

Checkpoint layout and distributed save rules are described under [Checkpoints](training.md#checkpoints) in the training guide.

## Distributed failures

**Reporter open on main.** If opening the reporter or logger fails on the main process, that rank sets Accelerate’s trigger; every rank observes the failure, destroys the process group, and the main re-raises the original error.

**During training.** Reporter errors on the main process are deferred and shared with other ranks on log intervals, after eval, and when leaving the loop. See [Halt](training.md#halt) in the training guide.

## See also

- [report API](api/report.md)
- [Training loop — Callbacks](training.md#callbacks)