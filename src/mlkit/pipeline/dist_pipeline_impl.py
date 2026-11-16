import logging
from typing import Any, Callable, List, Literal, Optional, cast

import datasets
import torch
from accelerate import PartialState, skip_first_batches
from overrides import override
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..util.decorators import inherit_docstrings
from ..util.iterator import inf_iterator_with_prefix
from .callback import BaseCallback, CallbackEvent
from .distributed_pipeline import (
    DistributedPipeline,
    DistributedTrainingConfig,
    LogConfig,
)


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Keep the numeric scalars of one metric payload as Python floats."""
    scalars: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        if isinstance(value, (int, float)):
            scalars[key] = float(value)
    return scalars


def _mean_metrics(samples: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric over the samples that report it."""
    values: dict[str, list[float]] = {}
    for sample in samples:
        for key, value in sample.items():
            values.setdefault(key, []).append(value)
    return {key: sum(vs) / len(vs) for key, vs in values.items()}


@inherit_docstrings
class MyTrainingConfig(DistributedTrainingConfig):
    def __init__(
        self,
        train_strategy: Literal["epochs", "steps"] = "epochs",
        n_epochs: int = 100,
        batch_size: int = 16,
        device: str = "cuda",
        save_strategy: Literal["epochs", "steps", None] = "epochs",
        save_dir: str | None = None,
        save_steps: int = 10,
        save_epochs: int = 1,
        save_total_limit: int = 5,
        num_workers: int = 4,
        train_shuffle: bool = True,
        eval_strategy: Literal["epochs", "steps"] | None = None,
        eval_steps: int = 500,
        eval_epochs: int = 1,
        n_steps: int | None = None,
        grad_clip_strategy: Literal["norm", "value", None] = "norm",
        max_grad_norm: float = 1.0,
        max_grad_value: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = "fp16",
        find_unused_parameters: bool = False,
        *_args,
        **kwargs,
    ):
        """Hydra-facing training config with epoch or step strategy and ``n_steps``.

        When ``real_batch_size`` is passed in ``kwargs``, gradient accumulation is
        recomputed from the live process count.
        """
        self.train_strategy = train_strategy
        real_batch_size = kwargs.get("real_batch_size", None)
        if real_batch_size is not None:
            print("real_batch_size is provided, so gradient_accumulation_steps is overridden")
            gradient_accumulation_steps = self.get_gradient_accumulation_steps(real_batch_size, batch_size)
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            save_strategy=save_strategy,
            save_dir=save_dir,
            save_steps=save_steps,
            save_epochs=save_epochs,
            save_total_limit=save_total_limit,
            num_workers=num_workers,
            train_shuffle=train_shuffle,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            eval_epochs=eval_epochs,
            grad_clip_strategy=grad_clip_strategy,
            max_grad_norm=max_grad_norm,
            max_grad_value=max_grad_value,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            find_unused_parameters=find_unused_parameters,
        )
        self.n_steps: int | None = n_steps
        skip_keys = ["real_batch_size"]
        for key, value in kwargs.items():
            if key in skip_keys:
                continue
            setattr(self, key, value)

    def get_gradient_accumulation_steps(self, real_batch_size, per_device_batch_size) -> int:
        """Get the gradient accumulation steps

        Args:
            real_batch_size: the real batch size
            per_device_batch_size: the batch size per device

        Returns:
            int: the gradient accumulation steps
        """
        n_processes = PartialState().num_processes
        assert real_batch_size % n_processes == 0, "real_batch_size must be divisible by n_processes"
        per_device_real_batch_size = real_batch_size // n_processes
        assert per_device_real_batch_size % per_device_batch_size == 0, (
            "per_device_real_batch_size must be divisible by per_device_batch_size"
        )
        gradient_accumulation_steps = per_device_real_batch_size // per_device_batch_size
        return gradient_accumulation_steps


@inherit_docstrings
class MyDistributedPipeline(DistributedPipeline):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset | datasets.Dataset,
        eval_dataset: torch.utils.data.Dataset | datasets.Dataset,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler],
        training_config: MyTrainingConfig,
        log_config: LogConfig,
        logger: logging.Logger | None,
        collate_fn: Callable | None = None,
        seed: int = 42,
        callbacks: Optional[List[BaseCallback]] = None,
    ):
        super().__init__(
            model=model,
            dataset=train_dataset,
            optimizers=optimizers,
            training_config=training_config,
            log_config=log_config,
            logger=logger,
            collate_fn=collate_fn,
            seed=seed,
            callbacks=callbacks,
        )
        self.eval_dataset = eval_dataset
        self.eval_dataloader = DataLoader(
            cast(torch.utils.data.Dataset, self.eval_dataset),
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
            collate_fn=collate_fn,
        )
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        self.training_config = cast(MyTrainingConfig, self.training_config)

    @override
    def compute_loss(
        self,
        model,
        batch: dict,
        *,
        phase: Literal["train", "eval"] = "train",
    ) -> dict[str, Any]:
        self.trigger_callbacks(event=CallbackEvent.PRE_COMPUTE_LOSS, batch=batch)
        loss = None
        model_output = model(**batch)
        if isinstance(model_output, dict):
            assert "loss" in model_output, "model_output must contain 'loss' key if model_output is a dict"
            loss = model_output["loss"]
        else:
            loss = model_output
        parsed_model_output = model_output if isinstance(model_output, dict) else None
        self.trigger_callbacks(
            event=CallbackEvent.POST_COMPUTE_LOSS,
            batch=batch,
            loss=loss,
            model_output=parsed_model_output,
            phase=phase,
            can_log=phase == "train" and self._log_interval_due("steps"),
        )
        return {
            "loss": loss,
            "model_output": parsed_model_output,
        }

    def eval_a_step(self, batch: dict) -> dict:
        """Evaluate the model for one step

        ``POST_EVAL_STEP`` callbacks see ``result["model_output"]``. The
        returned result keeps only ``eval_loss`` and the model's ``metrics``.

        Args:
            batch: the batch of data

        Returns:
            dict: ``eval_loss`` plus ``metrics`` when the model reports them
        """
        self.trigger_callbacks(event=CallbackEvent.PRE_EVAL_STEP, batch=batch)
        out = self.compute_loss(self.model, batch, phase="eval")
        model_output = out.get("model_output")
        result: dict[str, Any] = {"eval_loss": out["loss"].item()}
        if model_output is not None:
            result["model_output"] = model_output
        self.trigger_callbacks(
            event=CallbackEvent.POST_EVAL_STEP,
            batch=batch,
            result=result,
            loss=result["eval_loss"],
        )
        result.pop("model_output", None)
        metrics = model_output.get("metrics") if model_output is not None else None
        if isinstance(metrics, dict):
            result["metrics"] = metrics
        return result

    @override
    def train(self):
        training_config = cast(MyTrainingConfig, self.training_config)
        if training_config.train_strategy in ["epochs"]:
            return super().train()
        self.trigger_callbacks(event=CallbackEvent.TRAINING_START)
        if training_config.n_steps is None:
            raise ValueError("n_steps must be specified")

        num_batches_per_pass = len(self.dataloader)
        resume_offset = self.training_state.current_global_step % num_batches_per_pass
        first_pass_dataloader = skip_first_batches(self.dataloader, resume_offset) if resume_offset > 0 else None
        self.training_set_iterator = inf_iterator_with_prefix(self.dataloader, prefix_iterable=first_pass_dataloader)

        result = None
        remaining_steps = training_config.n_steps - self.training_state.current_global_step
        with tqdm(
            range(remaining_steps),
            desc="training",
            mininterval=0,
            initial=self.training_state.current_global_step,
            total=training_config.n_steps,
            disable=not self.accelerator.is_local_main_process,
        ) as pbar:
            for _ in pbar:
                batch = next(self.training_set_iterator)
                result = self.train_a_step(batch)
                self.training_state.current_step_in_epoch += 1
                self.training_state.current_global_step += 1

                if self._halt:
                    break
                if self._can_eval(flag="steps"):
                    self.eval()
                if self._halt:
                    break
                if self._can_save(flag="steps"):
                    self.save()
                if self._halt:
                    break
                pbar.set_postfix(loss=result["loss"], lr=result["lr"])
        self.save()
        self._sync_halt()
        self.trigger_callbacks(event=CallbackEvent.TRAINING_END)
        return result

    @torch.no_grad()
    def eval(self, disable_grad: bool = False):
        self.trigger_callbacks(event=CallbackEvent.PRE_EVAL, eval_dataloader=self.eval_dataloader)
        self.model.eval()
        eval_results = []
        gathered_losses = []
        gathered_metrics: list[dict[str, float]] = []
        for batch in self.eval_dataloader:
            if disable_grad:
                with torch.no_grad():
                    result = self.eval_a_step(batch)
            else:
                result = self.eval_a_step(batch)
            eval_results.append(result)

            # Replicate the batch scalar across the batch dim so gather_for_metrics
            # can trim padded duplicates on the final incomplete batch.
            batch_size = next(v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor))
            loss_tensor = torch.tensor(result["eval_loss"], device=self.accelerator.device).repeat(batch_size)
            gathered_losses.append(self.accelerator.gather_for_metrics(loss_tensor))
            sample_metrics = [_scalar_metrics(result.get("metrics", {}))] * batch_size
            gathered_metrics.extend(self.accelerator.gather_for_metrics(sample_metrics, use_gather_object=True))

        if len(gathered_losses) == 0:
            result = {
                "mean_eval_loss": float("nan"),
                "max_eval_loss": float("nan"),
                "min_eval_loss": float("nan"),
                "std_eval_loss": float("nan"),
            }
        else:
            all_losses = torch.cat(gathered_losses)
            result = {
                "mean_eval_loss": all_losses.mean().item(),
                "max_eval_loss": all_losses.max().item(),
                "min_eval_loss": all_losses.min().item(),
                "std_eval_loss": all_losses.std().item() if all_losses.numel() > 1 else 0.0,
            }
        result.update({f"eval/{key}": value for key, value in _mean_metrics(gathered_metrics).items()})
        self.log(result, step=self.training_state.current_global_step)
        self._sync_halt()

        self.model.train()
        self.trigger_callbacks(
            event=CallbackEvent.POST_EVAL,
            eval_dataloader=self.eval_dataloader,
            eval_results=eval_results,
        )
        return eval_results
