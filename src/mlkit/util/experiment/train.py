"""Training task for :func:`mlkit.util.experiment.run_experiment`.

Reads ``cfg.train`` (``seed``, ``mixed_precision``, schedule and save fields) and ``cfg.log``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from omegaconf import DictConfig

from mlkit.pipeline.callback import BaseCallback, CallbackEvent

from .runner import GetDatasetFn, GetModelFn, RunContext, Task, build_model_bundle

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from mlkit.pipeline.distributed_pipeline import DistributedPipeline
    from mlkit.pipeline.pipeline import LogConfig, TrainingConfig

CONFIG_SNAPSHOT_NAME = "config.yaml"


def checkpoint_root(cfg: DictConfig, suffix: str = "") -> str:
    """Return the directory holding the ``checkpoint_*`` folders of the training run.

    Args:
        cfg: experiment config; ``train.save_dir``, model, dataset, ``optimizer.lr``,
            and ``train.batch_size`` select the directory.
        suffix: the training task's ``run_suffix``.
    """
    from mlkit.util.utils_for_main import get_new_save_dir

    return get_new_save_dir(cfg.train.save_dir, cfg, suffix=suffix)


class ConfigSnapshotCallback(BaseCallback):
    """Write the resolved experiment config into each checkpoint before it is published.

    The config is read at save time, so rewrites made before training (LR scheduler
    length) are included.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def on_event(self, event: CallbackEvent, *_args: object, **kwargs: object) -> None:
        import os

        from omegaconf import OmegaConf

        if event is not CallbackEvent.PRE_FINALIZE_SAVE:
            return
        directory = kwargs.get("checkpoint_dir")
        if isinstance(directory, str):
            OmegaConf.save(self._cfg, os.path.join(directory, CONFIG_SNAPSHOT_NAME), resolve=True)


class GetCollateFn(Protocol):
    def __call__(self, cfg: DictConfig) -> Any: ...


class GetOptimizerFn(Protocol):
    def __call__(self, model: Any, cfg: DictConfig) -> Any: ...


@dataclass(kw_only=True)
class TrainRun(RunContext):
    """Training state built before ``pipeline.train()``.

    Teardown hooks receive the same instance after training; ``pipeline`` reflects
    post-train state.
    """

    model_bundle: dict[str, Any]
    train_set: Any
    val_set: Any
    test_set: Any
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    log_config: LogConfig
    training_config: TrainingConfig
    collate_fn: Callable[..., Any] | None
    pipeline: DistributedPipeline


@dataclass(kw_only=True)
class TrainSpec(Task[TrainRun]):
    """Train one model with the Accelerate pipeline.

    ``get_model`` may return a bare model or a bundle dict with ``"model"`` plus
    optional ``"callbacks"`` (passed to the pipeline constructor, so they see
    ``POST_LOAD``) and ``"train_hook_handlers"``. ``run_suffix`` is appended to
    both the run name and the checkpoint directory. Every published checkpoint
    also holds the resolved config as :data:`CONFIG_SNAPSHOT_NAME`.

    Hook timing (relative to ``accelerator.end_training()``):
        ``pre_dist_teardown``: every rank, after ``pipeline.train()`` while the
            process group is still alive (optional second training, shared reporter).
        ``post_dist_teardown``: global main only, after ``end_training``.
    """

    section: ClassVar[str] = "train"

    get_model: GetModelFn
    get_dataset: GetDatasetFn
    get_collate: GetCollateFn
    get_optimizer: GetOptimizerFn | None = None
    pre_dist_teardown: Callable[[TrainRun], None] | None = None
    post_dist_teardown: Callable[[TrainRun], None] | None = None
    run_suffix: str = ""
    disable_train_hooks: bool = True

    def run_name(self, cfg: DictConfig) -> str:
        from mlkit.util.utils_for_main import get_run_name

        return get_run_name(cfg, suffix=self.run_suffix)

    def resume_run_id(self, cfg: DictConfig, backend: str) -> str | None:
        from mlkit.pipeline.pipeline import latest_checkpoint_dir
        from mlkit.util.report import resume_run_id

        return resume_run_id(latest_checkpoint_dir(checkpoint_root(cfg, self.run_suffix)), backend)

    def run(self, ctx: RunContext) -> TrainRun:
        run = self._materialize(ctx)
        run.pipeline.train()
        if self.pre_dist_teardown is not None and not run.pipeline.halted:
            self.pre_dist_teardown(run)
        return run

    def finish(self, run: TrainRun) -> None:
        if self.post_dist_teardown is not None:
            self.post_dist_teardown(run)

    def _materialize(self, ctx: RunContext) -> TrainRun:
        """Build model, data, optimizer, scheduler, and pipeline (no ``pipeline.train()``).

        May rewrite ``cfg`` for the LR scheduler before ``store_config``.
        """
        from omegaconf import OmegaConf

        from mlkit.pipeline.pipeline import LogConfig
        from mlkit.util.show import show_info
        from mlkit.util.utils_for_main import (
            get_learing_rate_scheduler,
            get_optimizer,
            get_train_class,
        )

        cfg = ctx.cfg
        model_bundle = build_model_bundle(self.get_model, cfg, disable_train_hooks=self.disable_train_hooks)
        model = model_bundle["model"]

        train_set, val_set, test_set = self.get_dataset(cfg)
        optimizer = (self.get_optimizer or get_optimizer)(model, cfg)
        lr_scheduler = get_learing_rate_scheduler(optimizer, ctx.accelerator, train_set, cfg)
        if ctx.accelerator.is_main_process:
            effective_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            if ctx.reporter is not None:
                ctx.reporter.store_config(effective_config)
            if ctx.logger is not None:
                ctx.logger.info("Config after scheduler rewrite:\n" + OmegaConf.to_yaml(cfg))
        show_info(model=model, optimizer=optimizer)

        log_config = LogConfig(**cfg.log)
        PipelineClass, TrainingConfigClass = get_train_class()
        training_config = TrainingConfigClass(**cfg.train)
        training_config.save_dir = checkpoint_root(cfg, self.run_suffix)

        print(training_config.__dict__)

        collate_fn = self.get_collate(cfg)
        callbacks = [*model_bundle.get("callbacks", []), ConfigSnapshotCallback(cfg)]
        if ctx.reporter is not None:
            callbacks.append(ctx.reporter.as_callback())
        pipeline = PipelineClass(
            model=model,
            train_dataset=train_set,
            eval_dataset=val_set,
            optimizers=(optimizer, lr_scheduler),
            training_config=training_config,
            log_config=log_config,
            collate_fn=collate_fn,
            logger=ctx.logger,
            seed=cfg.train.seed,
            callbacks=callbacks,
        )
        return TrainRun(
            cfg=cfg,
            accelerator=ctx.accelerator,
            logger=ctx.logger,
            reporter=ctx.reporter,
            model_bundle=model_bundle,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            log_config=log_config,
            training_config=training_config,
            collate_fn=collate_fn,
            pipeline=pipeline,
        )
