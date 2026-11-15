from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from omegaconf import DictConfig

from mlkit.util.report.base import Reporter


class GetModelFn(Protocol):
    def __call__(self, cfg: DictConfig) -> Any: ...


class GetDatasetFn(Protocol):
    def __call__(self, cfg: DictConfig) -> tuple[Any, Any, Any]: ...


class GetCollateFn(Protocol):
    def __call__(self, cfg: DictConfig) -> Any: ...


class GetOptimizerFn(Protocol):
    def __call__(self, model: Any, cfg: DictConfig) -> Any: ...


@dataclass
class MaterializedRun:
    """Primary training run state from :func:`mlkit.util.experiment.run_experiment`.

    Filled during materialize (before ``pipeline.train()``). Teardown hooks receive the
    same instance after training; ``pipeline`` reflects post-train state.
    """

    cfg: DictConfig
    accelerator: Any
    reporter: Reporter | None
    model_bundle: dict[str, Any]
    train_set: Any
    val_set: Any
    test_set: Any
    optimizer: Any
    lr_scheduler: Any
    log_config: Any
    training_config: Any
    collate_fn: Any
    pipeline: Any


@dataclass
class ExperimentSpec:
    """Specification for :func:`mlkit.util.experiment.run_experiment`.

    ``get_model`` may return a bare model or a bundle dict with ``"model"`` plus
    optional ``"callbacks"`` (passed to the pipeline constructor, so they see
    ``POST_LOAD``) and ``"train_hook_handlers"``. ``run_suffix`` is appended to
    both the run name and the checkpoint directory.

    Hook timing (relative to ``accelerator.end_training()``):
        ``pre_dist_teardown``: every rank, after ``pipeline.train()`` while the
            process group is still alive (optional second training, shared reporter).
        ``post_dist_teardown``: global main only, after ``end_training`` (sampling,
            plots; single-process work).
    """

    get_model: GetModelFn
    get_dataset: GetDatasetFn
    get_collate: GetCollateFn
    get_optimizer: GetOptimizerFn | None = None
    pre_dist_teardown: Callable[[MaterializedRun], None] | None = None
    post_dist_teardown: Callable[[MaterializedRun], None] | None = None
    run_suffix: str = ""
    disable_train_hooks: bool = True
