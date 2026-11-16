"""Task-agnostic Hydra/OmegaConf entry shared by training, inference, and custom tasks.

Expects ``cfg`` sections ``log``, ``reporter``, ``mlflow``, and ``wandb``, plus the
section named by :attr:`Task.section` with ``seed`` and ``mixed_precision``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from omegaconf import DictConfig

if TYPE_CHECKING:
    import logging

    from accelerate import Accelerator

    from mlkit.util.report.base import Reporter


class GetModelFn(Protocol):
    def __call__(self, cfg: DictConfig) -> Any: ...


class GetDatasetFn(Protocol):
    def __call__(self, cfg: DictConfig) -> tuple[Any, Any, Any]: ...


@dataclass(kw_only=True)
class RunContext:
    """Process state that :func:`run_experiment` hands to a :class:`Task`.

    ``logger`` and ``reporter`` are ``None`` off the global main process.
    """

    cfg: DictConfig
    accelerator: Accelerator
    logger: logging.Logger | None
    reporter: Reporter | None


class Task[R: RunContext](ABC):
    """One job that :func:`run_experiment` runs inside a single process group.

    ``cfg[section]`` supplies ``seed`` and ``mixed_precision``.

    Hook timing (relative to :func:`teardown_process_group`):
        ``run``: every rank, while the process group is alive.
        ``finish``: global main only, after process group teardown, while the
            reporter is still open.
    """

    section: ClassVar[str]

    @abstractmethod
    def run_name(self, cfg: DictConfig) -> str:
        """Return the tracked run name. Called on the global main only."""

    def resume_run_id(self, cfg: DictConfig, backend: str) -> str | None:
        """Return the tracked run to continue, or ``None`` for a new run.

        Args:
            cfg: experiment config.
            backend: reporter that is about to be opened.
        """
        del cfg, backend
        return None

    @abstractmethod
    def run(self, ctx: RunContext) -> R:
        """Do the work on every rank and return the state handed to :meth:`finish`."""

    def finish(self, run: R) -> None:
        """Do single-process work on the global main after the process group is gone."""
        del run


def teardown_process_group(
    accelerator: Accelerator,
    *,
    logger: logging.Logger | None = None,
    wait: bool = True,
) -> None:
    """Synchronize ranks (optional) and destroy the Accelerate process group.

    Wraps ``accelerator.end_training()`` so training and inference teardown share
    one call site if Accelerate renames or splits this API.

    Args:
        accelerator: Accelerate instance for this run.
        logger: file logger on the global main; used when teardown raises.
        wait: when true, call ``wait_for_everyone`` before destroying the group.
    """
    if wait:
        accelerator.wait_for_everyone()
    try:
        accelerator.end_training()
    except Exception:
        if logger is not None:
            logger.exception("Failed to destroy the process group")


def run_experiment[R: RunContext](cfg: DictConfig, task: Task[R]) -> R:
    """Run ``task`` inside one Accelerate process group with logging and tracking.

    The global main opens the log file and the reporter first; a failure there stops
    every rank before ``task.run``. A failure inside ``task.run`` calls
    :func:`teardown_process_group` without a barrier and skips ``task.finish``.
    On success, :func:`teardown_process_group` runs with a barrier, a deferred
    rank-local failure may be raised, and ``task.finish`` runs on the global main.

    Returns:
        The state returned by ``task.run`` on this rank.
    """

    import os
    from contextlib import ExitStack

    from accelerate import Accelerator
    from omegaconf import OmegaConf

    from mlkit.util.failure import clear_deferred_error, raise_deferred_error
    from mlkit.util.log import get_and_create_new_log_dir, get_logger
    from mlkit.util.report import ReporterContext
    from mlkit.util.seed import seed_everything

    settings = cfg[task.section]
    seed_everything(settings.seed)

    accelerator = Accelerator(mixed_precision=settings.mixed_precision)
    print(f"accelerator.device = {accelerator.device}")

    logger: logging.Logger | None = None
    reporter: Reporter | None = None

    def _reporter_open_failed(local_open_error: BaseException | None) -> bool:
        if accelerator.num_processes == 1:
            return local_open_error is not None
        if local_open_error is not None:
            accelerator.set_trigger()
        return bool(accelerator.check_trigger())

    clear_deferred_error()
    with ExitStack() as stack:
        local_open_error: BaseException | None = None
        if accelerator.is_main_process:
            try:
                log_dir = get_and_create_new_log_dir(cfg.log.log_dir, suffix=str(os.getpid()))
                logger = get_logger(name="experiment", log_dir=log_dir)
                logger.info(f"accelerator.device = {accelerator.device}")
                logger.info(f"seed = {settings.seed}")
                logger.info("Config:\n" + OmegaConf.to_yaml(cfg))
                reporter_kind = str(cfg.reporter).strip().lower()
                reporter_context = ReporterContext(
                    reporter_kind,
                    run_name=task.run_name(cfg),
                    config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
                    mlflow=cast(dict[str, Any], OmegaConf.to_container(cfg.mlflow, resolve=True)),
                    wandb=cast(dict[str, Any], OmegaConf.to_container(cfg.wandb, resolve=True)),
                    resume_run_id=task.resume_run_id(cfg, reporter_kind),
                )
                reporter = stack.enter_context(reporter_context)
            except BaseException as exc:
                local_open_error = exc
        if _reporter_open_failed(local_open_error):
            teardown_process_group(accelerator, logger=logger, wait=False)
            if local_open_error is not None:
                raise local_open_error
            raise RuntimeError("Reporter failed to open on another rank")
        ctx = RunContext(cfg=cfg, accelerator=accelerator, logger=logger, reporter=reporter)
        try:
            run = task.run(ctx)
        except BaseException:
            teardown_process_group(accelerator, logger=logger, wait=False)
            raise
        teardown_process_group(accelerator, logger=logger, wait=True)
        raise_deferred_error()
        if accelerator.is_main_process:
            task.finish(run)
    return run
