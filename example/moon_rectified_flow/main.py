"""Hydra training example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from utils import get_collate_fn, get_dataset, get_model

from mlkit.util.experiment import ConfigSnapshotCallback, TrainSpec, run_experiment


def after_train(run):
    """Train the conditional flow on every rank while the process group is alive."""
    cfg = run.cfg
    accelerator = run.accelerator
    del run.model_bundle, run.pipeline, run.train_set
    from typing import Any, cast

    from omegaconf import OmegaConf
    from shared.dataset import MoonsDataset
    from utils import get_conditional_collate_fn, get_conditional_model, get_conditional_save_dir

    from mlkit.pipeline.pipeline import LogConfig
    from mlkit.util.log import get_and_create_new_log_dir, get_logger
    from mlkit.util.utils_for_main import (
        get_learing_rate_scheduler,
        get_optimizer,
        get_train_class,
    )

    cond_result = get_conditional_model(cfg)
    cond_pipeline_model = cond_result["model"]

    cond_dataset = MoonsDataset(n_samples=256)
    cond_collate = get_conditional_collate_fn(cfg)

    cond_opt = get_optimizer(cond_pipeline_model, cfg)
    cond_sched = get_learing_rate_scheduler(cond_opt, accelerator, cond_dataset, cfg)

    PipelineClass, TrainingConfigClass = get_train_class()
    train_kwargs: dict[str, Any] = cast(dict[str, Any], OmegaConf.to_container(cfg.train, resolve=True))
    train_kwargs["save_dir"] = get_conditional_save_dir(cfg)
    cond_training_config = TrainingConfigClass(**train_kwargs)
    log_kwargs: dict[str, Any] = cast(dict[str, Any], OmegaConf.to_container(cfg.log, resolve=True))
    log_config = LogConfig(**log_kwargs)

    logger = None
    if accelerator.is_main_process:
        logger = get_logger(name="cond_experiment", log_dir=get_and_create_new_log_dir(cfg.log.log_dir))

    cond_pipeline = PipelineClass(
        model=cond_pipeline_model,
        train_dataset=cond_dataset,
        eval_dataset=cond_dataset,
        optimizers=(cond_opt, cond_sched),
        training_config=cond_training_config,
        log_config=log_config,
        logger=logger,
        collate_fn=cond_collate,
        callbacks=[ConfigSnapshotCallback(cfg)],
    )
    if run.reporter is not None:
        cond_pipeline.add_callbacks([run.reporter.as_callback()])
    cond_pipeline.train()


SPEC = TrainSpec(
    get_model=get_model,
    get_dataset=get_dataset,
    get_collate=get_collate_fn,
    after_train=after_train,
    disable_train_hooks=True,
)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_experiment(cfg, SPEC)


if __name__ == "__main__":
    main()
