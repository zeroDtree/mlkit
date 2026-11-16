"""Inference task for :func:`mlkit.util.experiment.run_experiment`.

Reads ``cfg.infer`` (``seed``, ``mixed_precision``, ``checkpoint_dir``). The training
sections locate the default checkpoint and name the run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from omegaconf import DictConfig

from .runner import GetDatasetFn, GetModelFn, RunContext, Task


@dataclass(kw_only=True)
class InferRun(RunContext):
    """Inference state with trained weights loaded.

    ``model_bundle["model"]`` is on ``accelerator.device`` in eval mode. The datasets
    are ``None`` without ``get_dataset``. ``outputs`` holds what ``infer`` returned.
    """

    model_bundle: dict[str, Any]
    checkpoint_dir: str
    train_set: Any = None
    val_set: Any = None
    test_set: Any = None
    outputs: Any = None


@dataclass(kw_only=True)
class InferSpec(Task[InferRun]):
    """Load trained weights, then run inference hooks.

    Weights come from ``cfg.infer.checkpoint_dir`` when set, otherwise from the
    latest checkpoint under the training save directory for ``run_suffix``. Use the
    same ``run_suffix`` and training overrides as the training run. ``get_model``
    builds the model as for training; its ``"train_hook_handlers"`` are disabled.

    Hook timing (relative to :func:`teardown_process_group`):
        ``infer``: every rank, while the process group is alive (sharded inference,
            collectives). Its return value is stored in ``run.outputs``.
        ``post_dist_teardown``: global main only, after process group teardown (sampling,
            plots; single-process work). Invoked from :meth:`Task.finish`.
    """

    section: ClassVar[str] = "infer"

    get_model: GetModelFn
    get_dataset: GetDatasetFn | None = None
    infer: Callable[[InferRun], Any] | None = None
    post_dist_teardown: Callable[[InferRun], None] | None = None
    run_suffix: str = ""

    def run_name(self, cfg: DictConfig) -> str:
        from mlkit.util.utils_for_main import get_run_name

        return get_run_name(cfg, suffix=f"{self.run_suffix}-infer")

    def run(self, ctx: RunContext) -> InferRun:
        import os

        from accelerate.utils import SAFE_WEIGHTS_NAME
        from safetensors.torch import load_model

        cfg = ctx.cfg
        checkpoint_dir = self._checkpoint_dir(cfg)
        model_bundle = self.get_model(cfg)
        if not isinstance(model_bundle, dict):
            model_bundle = {"model": model_bundle}
        for handler in model_bundle.get("train_hook_handlers", []):
            handler.disable()

        model = model_bundle["model"]
        load_model(model, os.path.join(checkpoint_dir, SAFE_WEIGHTS_NAME))
        model.to(ctx.accelerator.device).eval()

        datasets = self.get_dataset(cfg) if self.get_dataset is not None else (None, None, None)
        train_set, val_set, test_set = datasets
        run = InferRun(
            cfg=cfg,
            accelerator=ctx.accelerator,
            logger=ctx.logger,
            reporter=ctx.reporter,
            model_bundle=model_bundle,
            checkpoint_dir=checkpoint_dir,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
        )
        if self.infer is not None:
            run.outputs = self.infer(run)
        return run

    def finish(self, run: InferRun) -> None:
        if self.post_dist_teardown is not None:
            self.post_dist_teardown(run)

    def _checkpoint_dir(self, cfg: DictConfig) -> str:
        from mlkit.pipeline.pipeline import latest_checkpoint_dir
        from mlkit.util.utils_for_main import get_new_save_dir

        explicit = cfg.infer.get("checkpoint_dir")
        if explicit:
            return str(explicit)
        save_dir = get_new_save_dir(cfg.train.save_dir, cfg, suffix=self.run_suffix)
        checkpoint_dir = latest_checkpoint_dir(save_dir)
        if checkpoint_dir is None:
            raise FileNotFoundError(f"No checkpoint under {save_dir}; train first or set infer.checkpoint_dir")
        return checkpoint_dir
