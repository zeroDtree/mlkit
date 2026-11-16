"""Inference task for :func:`mlkit.util.experiment.run_experiment`.

Reads ``cfg.infer`` (``seed``, ``mixed_precision``, ``checkpoint_dir``). The training
sections locate the default checkpoint and name the run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from omegaconf import DictConfig

from .runner import GetDatasetFn, GetModelFn, RunContext, Task, build_model_bundle
from .train import CONFIG_SNAPSHOT_NAME, checkpoint_root

if TYPE_CHECKING:
    import logging

_RUNTIME_SECTIONS = frozenset({"train", "infer", "optimizer", "log", "reporter", "mlflow", "wandb", "hydra"})


@dataclass(kw_only=True)
class InferRun(RunContext):
    """Inference state with trained weights loaded.

    ``model_bundle["model"]`` is on ``accelerator.device`` in eval mode. ``train_cfg``
    is the config saved with the checkpoint, or ``None`` for checkpoints without one.
    The datasets are ``None`` without ``get_dataset``. ``outputs`` holds what
    ``infer`` returned.
    """

    model_bundle: dict[str, Any]
    checkpoint_dir: str
    train_cfg: DictConfig | None = None
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
    The global main logs a warning for every model-side config value that differs
    from the config saved with the checkpoint.

    Hook timing (relative to ``accelerator.end_training()``):
        ``infer``: every rank, while the process group is alive (sharded inference,
            collectives). Its return value is stored in ``run.outputs``.
        ``post_dist_teardown``: global main only, after ``end_training`` (sampling,
            plots; single-process work).

    Both hooks run under ``accelerator.autocast()``, so ``infer.mixed_precision``
    applies to them.
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
        train_cfg = _load_train_cfg(checkpoint_dir)
        if ctx.logger is not None:
            _warn_on_config_drift(ctx.logger, checkpoint_dir, train_cfg, cfg)

        model_bundle = build_model_bundle(self.get_model, cfg, disable_train_hooks=True)
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
            train_cfg=train_cfg,
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
        )
        if self.infer is not None:
            with ctx.accelerator.autocast():
                run.outputs = self.infer(run)
        return run

    def finish(self, run: InferRun) -> None:
        if self.post_dist_teardown is not None:
            with run.accelerator.autocast():
                self.post_dist_teardown(run)

    def _checkpoint_dir(self, cfg: DictConfig) -> str:
        from mlkit.pipeline.pipeline import latest_checkpoint_dir

        explicit = cfg.infer.get("checkpoint_dir")
        if explicit:
            return str(explicit)
        save_dir = checkpoint_root(cfg, self.run_suffix)
        checkpoint_dir = latest_checkpoint_dir(save_dir)
        if checkpoint_dir is None:
            raise FileNotFoundError(f"No checkpoint under {save_dir}; train first or set infer.checkpoint_dir")
        return checkpoint_dir


def _load_train_cfg(checkpoint_dir: str) -> DictConfig | None:
    import os

    from omegaconf import OmegaConf

    path = os.path.join(checkpoint_dir, CONFIG_SNAPSHOT_NAME)
    if not os.path.exists(path):
        return None
    train_cfg = OmegaConf.load(path)
    if not isinstance(train_cfg, DictConfig):
        raise TypeError(f"{path} does not hold a mapping")
    return train_cfg


def _warn_on_config_drift(
    logger: logging.Logger, checkpoint_dir: str, train_cfg: DictConfig | None, cfg: DictConfig
) -> None:
    if train_cfg is None:
        logger.warning(f"{checkpoint_dir} has no {CONFIG_SNAPSHOT_NAME}; the config check is skipped")
        return
    drift = _config_drift(train_cfg, cfg)
    if drift:
        logger.warning("Inference config differs from the training config:\n" + "\n".join(drift))


def _config_drift(train_cfg: DictConfig, cfg: DictConfig) -> list[str]:
    """List model-side values that differ between the training and inference configs.

    Runtime sections (``train``, ``infer``, ``optimizer``, ``log``, reporter backends)
    are ignored. Sampling-only values that you change on purpose are listed too.

    Returns:
        One ``key: train=..., infer=...`` line per differing dotted key, sorted.
    """
    from omegaconf import OmegaConf

    trained = _flatten(OmegaConf.to_container(train_cfg, resolve=True))
    current = _flatten(OmegaConf.to_container(cfg, resolve=True))
    return [
        f"{key}: train={trained.get(key, '<unset>')!r}, infer={current.get(key, '<unset>')!r}"
        for key in sorted(trained.keys() | current.keys())
        if key.partition(".")[0] not in _RUNTIME_SECTIONS
        and (key not in trained or key not in current or trained[key] != current[key])
    ]


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(node, dict):
        return {prefix: node}
    flat: dict[str, Any] = {}
    for key, value in node.items():
        flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat
