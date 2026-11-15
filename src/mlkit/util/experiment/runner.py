"""Generic Hydra/OmegaConf training entry for examples.

Expects ``cfg`` sections such as ``train``, ``log``, ``reporter``, ``mlflow``, and ``wandb``.
"""

from __future__ import annotations

from typing import Any, cast

from omegaconf import DictConfig

from .types import ExperimentSpec, TrainRunContext


def run_experiment(cfg: DictConfig, spec: ExperimentSpec) -> None:
    """Run a complete train cycle with optional pre/post distributed teardown hooks."""

    from contextlib import ExitStack

    from accelerate import Accelerator
    from omegaconf import OmegaConf

    from mlkit.pipeline.pipeline import LogConfig, latest_checkpoint_dir
    from mlkit.util.failure import clear_deferred_error, raise_deferred_error
    from mlkit.util.log import get_and_create_new_log_dir, get_logger
    from mlkit.util.report import ReporterContext, resume_run_id
    from mlkit.util.seed import seed_everything
    from mlkit.util.show import show_info
    from mlkit.util.utils_for_main import (
        get_learing_rate_scheduler,
        get_new_save_dir,
        get_optimizer,
        get_run_name,
        get_train_class,
    )

    seed_everything(cfg.train.seed)

    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
    print(f"accelerator.device = {accelerator.device}")

    logger = None
    run_name = ""
    reporter = None

    def _train():
        model_bundle = spec.get_model(cfg)
        if not isinstance(model_bundle, dict):
            model_bundle = {"model": model_bundle}
        model = model_bundle["model"]

        if spec.disable_train_hooks:
            for handler in model_bundle.get("train_hook_handlers", []):
                handler.disable()

        train_set, val_set, _ = spec.get_dataset(cfg)
        optimizer = (spec.get_optimizer or get_optimizer)(model, cfg)
        lr_scheduler = get_learing_rate_scheduler(optimizer, accelerator, train_set, cfg)
        if accelerator.is_main_process:
            effective_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            if reporter is not None:
                reporter.store_config(effective_config)
            if logger is not None:
                logger.info("Config after scheduler rewrite:\n" + OmegaConf.to_yaml(cfg))
        show_info(model=model, optimizer=optimizer)

        log_config = LogConfig(**cfg.log)
        PipelineClass, TrainingConfigClass = get_train_class()
        training_config = TrainingConfigClass(**cfg.train)
        training_config.save_dir = get_new_save_dir(training_config.save_dir, cfg, suffix=spec.save_dir_suffix)

        print(training_config.__dict__)

        pipeline = PipelineClass(
            model=model,
            train_dataset=train_set,
            eval_dataset=val_set,
            optimizers=(optimizer, lr_scheduler),
            training_config=training_config,
            log_config=log_config,
            collate_fn=spec.get_collate(cfg),
            logger=logger,
        )
        if reporter is not None:
            pipeline.add_callbacks([reporter.as_callback()])
        pipeline.train()
        return model_bundle, pipeline, train_set

    def _run_train() -> tuple[Any, Any, Any]:
        trained = _train()
        model_bundle, pipeline, train_set = trained
        if spec.pre_dist_teardown is not None and not pipeline._halt:
            ctx = TrainRunContext(
                cfg=cfg,
                model_bundle=model_bundle,
                pipeline=pipeline,
                accelerator=accelerator,
                train_set=train_set,
                reporter=reporter,
            )
            spec.pre_dist_teardown(ctx)
        return trained

    def _destroy_group() -> None:
        try:
            accelerator.end_training()
        except Exception:
            if logger is not None:
                logger.exception("Failed to destroy the process group")

    def _finish(trained: tuple[Any, Any, Any]) -> None:
        model_bundle, pipeline, train_set = trained
        accelerator.wait_for_everyone()
        accelerator.end_training()
        raise_deferred_error()
        if accelerator.is_main_process and spec.post_dist_teardown is not None:
            ctx = TrainRunContext(
                cfg=cfg,
                model_bundle=model_bundle,
                pipeline=pipeline,
                accelerator=accelerator,
                train_set=train_set,
                reporter=reporter,
            )
            spec.post_dist_teardown(ctx)

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
                log_dir = get_and_create_new_log_dir(cfg.log.log_dir)
                logger = get_logger(name="experiment", log_dir=log_dir)
                logger.info(f"accelerator.device = {accelerator.device}")
                logger.info(f"seed = {cfg.train.seed}")
                run_name = get_run_name(cfg)
                logger.info("Config:\n" + OmegaConf.to_yaml(cfg))
                resolved_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
                reporter_kind = str(cfg.reporter).strip().lower()
                checkpoint_dir = latest_checkpoint_dir(
                    get_new_save_dir(cfg.train.save_dir, cfg, suffix=spec.save_dir_suffix)
                )
                reporter_context = ReporterContext(
                    reporter_kind,
                    run_name=run_name,
                    config=resolved_config,
                    mlflow=cast(dict[str, Any], OmegaConf.to_container(cfg.mlflow, resolve=True)),
                    wandb=cast(dict[str, Any], OmegaConf.to_container(cfg.wandb, resolve=True)),
                    resume_run_id=resume_run_id(checkpoint_dir, reporter_kind),
                )
                reporter = stack.enter_context(reporter_context)
            except BaseException as exc:
                local_open_error = exc
        if _reporter_open_failed(local_open_error):
            _destroy_group()
            if local_open_error is not None:
                raise local_open_error
            raise RuntimeError("Reporter failed to open on another rank")
        try:
            trained = _run_train()
        except BaseException:
            _destroy_group()
            raise
        _finish(trained)