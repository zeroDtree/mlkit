import os

from accelerate import Accelerator
from omegaconf import DictConfig

from mlkit.util.training_steps import get_total_training_steps


def get_run_name(cfg: DictConfig, prefix: str = "", suffix: str = "") -> str:
    run_name = f"{cfg.dataset.name}-{cfg.model.name.replace('/', '_').replace('-', '_')}-lr:{cfg.optimizer.lr}"
    if cfg.train.train_strategy in ["steps"]:
        run_name += f"-n_steps:{cfg.train.n_steps}"
    if cfg.train.train_strategy in ["epochs"]:
        run_name += f"-n_epochs:{cfg.train.n_epochs}"
    run_name = f"{prefix}{run_name}{suffix}"
    return run_name


def get_optimizer(model, cfg: DictConfig):
    import torch  # noqa: F401

    optimizer_config = dict(cfg.optimizer)
    optimizer_class_name = optimizer_config.pop("name")
    optimizer = eval("torch.optim." + optimizer_class_name)(model.parameters(), **optimizer_config)
    return optimizer


def get_learing_rate_scheduler(
    optimizer,
    accelerator: Accelerator,
    train_set,
    cfg: DictConfig,
    inplace: bool = True,
):
    from mlkit.scheduler.lr_scheduler_factory import get_lr_scheduler

    cfg.train.batch_size
    real_batch_size = cfg.train.get("real_batch_size", None)
    if real_batch_size is None:
        real_batch_size = cfg.train.batch_size * cfg.train.gradient_accumulation_steps * accelerator.num_processes
    effective_batch_size = accelerator.num_processes * cfg.train.batch_size
    assert real_batch_size % effective_batch_size == 0, "real_batch_size must be divisible by effective_batch_size"
    gradient_accumulation_steps = real_batch_size // effective_batch_size

    n_training_steps = get_total_training_steps(
        train_strategy=cfg.train.train_strategy,
        dataset_len=len(train_set),
        n_epochs=cfg.train.n_epochs,
        n_steps=cfg.train.n_steps,
        num_processes=accelerator.num_processes,
        batch_size=cfg.train.batch_size,
    )
    if inplace:
        cfg.train.n_steps = n_training_steps
        cfg.train.real_batch_size = real_batch_size
        cfg.train.gradient_accumulation_steps = gradient_accumulation_steps

    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        n_warmup_steps=cfg.train.n_warmup_steps,
        n_training_steps=(n_training_steps * accelerator.num_processes),
        lr_scheduler_type=cfg.train.lr_scheduler_type,
    )
    return lr_scheduler


def get_train_class():
    from mlkit.pipeline.dist_pipeline_impl import (
        MyDistributedPipeline,
        MyTrainingConfig,
    )

    return MyDistributedPipeline, MyTrainingConfig


def get_new_save_dir(save_dir, cfg: DictConfig, prefix: str = "", suffix: str = ""):
    if save_dir is None:
        save_dir = "checkpoints"
    model_name = str(cfg.model.name.replace("/", "_").replace("-", "_"))
    dataset_name = str(cfg.dataset.name)
    lr = str(cfg.optimizer.lr)
    batch_size = str(cfg.train.batch_size)
    new_save_dir = os.path.join(save_dir, f"{model_name}/{dataset_name}/-lr:{lr}-bs:{batch_size}")
    new_save_dir = f"{prefix}{new_save_dir}{suffix}"
    return new_save_dir
