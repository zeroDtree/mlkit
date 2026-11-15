from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from omegaconf import DictConfig

from mlkit.util.report.base import Reporter


class GetModelFn(Protocol):
    def __call__(self, cfg: DictConfig, **kwargs: Any) -> Any: ...


class GetDatasetFn(Protocol):
    def __call__(self, cfg: DictConfig) -> tuple[Any, Any, Any]: ...


class GetCollateFn(Protocol):
    def __call__(self, cfg: DictConfig) -> Any: ...


class GetOptimizerFn(Protocol):
    def __call__(self, model: Any, cfg: DictConfig) -> Any: ...


@dataclass
class ExperimentSpec:
    """Specification for :func:`mlkit.util.experiment.run_experiment`.

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
    pre_dist_teardown: Callable[[TrainRunContext], None] | None = None
    post_dist_teardown: Callable[[TrainRunContext], None] | None = None
    save_dir_suffix: str = ""
    disable_train_hooks: bool = True


@dataclass
class TrainRunContext:
    """Hook context; ``model_bundle`` is the dict from ``get_model`` (at least a ``model`` key)."""

    cfg: DictConfig
    model_bundle: dict[str, Any]
    pipeline: Any
    accelerator: Any
    train_set: Any
    reporter: Reporter | None
