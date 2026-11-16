import logging
import os
import re
import shutil
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import datasets
import torch
from accelerate import skip_first_batches
from torch import Tensor
from tqdm.auto import tqdm

from ..util.failure import defer_error, has_deferred_error
from .callback import BaseCallback, CallbackEvent, CallbackManager


def _format_log(metrics: Mapping[str, Any], step: int | None) -> str:
    body = " ".join(f"{key}={value}" for key, value in metrics.items())
    if step is None:
        return body
    return f"step={step} {body}"


class TrainingConfig:
    def __init__(
        self,
        n_epochs: int = 100,
        batch_size: int = 4,
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
        grad_clip_strategy: Literal["norm", "value", None] = "norm",
        max_grad_norm: float = 1.0,
        max_grad_value: float = 1.0,
        gradient_accumulation_steps: int = 1,
        *_args,
        **_kwargs,
    ):
        """Training hyperparameters and checkpoint, eval, and save scheduling.

        See the Training loop guide (``docs/training.md``) for how each field drives
        the epoch and step clocks.
        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.save_strategy = save_strategy
        self.save_dir = save_dir
        self.save_steps = save_steps
        self.save_epochs = save_epochs
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle
        self.save_total_limit = save_total_limit
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.eval_epochs = eval_epochs
        self.grad_clip_strategy = grad_clip_strategy
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value
        self.gradient_accumulation_steps = gradient_accumulation_steps


class LogConfig:
    def __init__(
        self,
        log_dir: str = "logs",
        log_steps: int = 100,
        log_epochs: int = 1,
        log_strategy: Literal["epochs", "steps"] = "epochs",
        *_args,
        **_kwargs,
    ):
        """Directory and interval for file logging during training."""
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.log_epochs = log_epochs
        self.log_strategy = log_strategy


class TrainingState:
    def __init__(
        self,
        current_epoch: int = 0,
        current_step_in_epoch: int = 0,
        current_global_step: int = 0,
    ):
        """Epoch, in-epoch step, and global step counters for resume and intervals."""
        self.current_epoch = current_epoch
        self.current_step_in_epoch = current_step_in_epoch
        self.current_global_step = current_global_step


def latest_checkpoint_dir(save_dir: str | None) -> str | None:
    """Return the checkpoint directory with the largest ``global`` step.

    Args:
        save_dir: directory that contains ``checkpoint_*`` folders.

    Returns:
        The latest checkpoint directory, or ``None`` when none exist.
    """
    if save_dir is None or save_dir == "" or not os.path.exists(save_dir) or len(os.listdir(save_dir)) <= 0:
        return None

    def _checkpoint_sort_key(name: str) -> int:
        match = re.search(r"global(\d+)", name)
        return int(match.group(1)) if match is not None else 0

    checkpoints = [name for name in os.listdir(save_dir) if name.startswith("checkpoint_")]
    checkpoints.sort(key=_checkpoint_sort_key, reverse=True)
    if len(checkpoints) <= 0:
        return None
    return os.path.join(save_dir, checkpoints[0])


class BasePipeline(metaclass=ABCMeta):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler],
        training_config: TrainingConfig,
        log_config: LogConfig,
        logger: logging.Logger | None,
        collate_fn: Optional[Callable] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        load_checkpoint: bool = True,
        *_args,
        **_kwargs,
    ):
        """Wire model, data, optimizers, configs, callbacks, and optional checkpoint resume."""
        super().__init__()
        # callbacks
        self.callback_manager = CallbackManager()
        self.callback_manager.add_callbacks(callbacks=callbacks or [])

        # model, dataset, optimizers, log, training,
        self.model = model
        self.dataset = dataset
        self.optimizer, self.scheduler = optimizers
        self.training_config = training_config
        self.log_config = log_config
        self.training_state = TrainingState()
        self._halt = False
        self.logger = logger.getChild(type(self).__name__) if logger is not None else None

        self.dataloader = torch.utils.data.DataLoader(
            cast(torch.utils.data.Dataset, self.dataset),
            batch_size=self.training_config.batch_size,
            shuffle=self.training_config.train_shuffle,
            collate_fn=collate_fn,
            num_workers=self.training_config.num_workers,
        )

        if load_checkpoint and self.training_config.save_dir is not None and self.training_config.save_dir != "":
            self.load()
            self.training_config = training_config

    @abstractmethod
    def compute_loss(self, model: torch.nn.Module, batch: dict) -> Tensor | dict:
        """Compute loss for one batch."""

    def train(self) -> None:
        """Run the epoch-based training loop until ``n_epochs`` or halt."""
        self.trigger_callbacks(event=CallbackEvent.TRAINING_START)
        n_epochs = self.training_config.n_epochs

        for epoch in tqdm(
            range(n_epochs),
            desc="training",
            mininterval=0,
            disable=not self._is_local_main_process(),
        ):
            if epoch < self.training_state.current_epoch:
                continue
            completed = self.train_an_epoch()
            if completed:
                self.training_state.current_epoch += 1

            if self._halt:
                break
            if self._can_eval(flag="epochs"):
                self.eval()
            if self._halt:
                break
            if self._can_save(flag="epochs"):
                self.save()
            if self._halt:
                break
        self.save()
        self._sync_halt()
        self.trigger_callbacks(event=CallbackEvent.TRAINING_END)

    def train_an_epoch(self) -> bool:
        """Train one epoch.

        Returns:
            True when every step in the epoch ran. A halt after that still
            counts as complete. A halt inside the step loop does not.
        """
        self.trigger_callbacks(event=CallbackEvent.EPOCH_START)
        resume_step = self.training_state.current_step_in_epoch
        active_dataloader = skip_first_batches(self.dataloader, resume_step) if resume_step > 0 else self.dataloader
        result = None
        completed = True
        with tqdm(
            active_dataloader,
            desc=f"epoch {self.training_state.current_epoch}",
            total=len(self.dataloader),
            initial=resume_step,
            position=1,
            leave=False,
            mininterval=0,
            disable=not self._is_local_main_process(),
        ) as pbar:
            for batch in pbar:
                result = self.train_a_step(batch)
                self.training_state.current_step_in_epoch += 1
                self.training_state.current_global_step += 1

                if self._halt:
                    completed = False
                    break
                if self._can_eval(flag="steps"):
                    self.eval()
                if self._halt:
                    completed = False
                    break
                if self._can_save(flag="steps"):
                    self.save()
                if self._halt:
                    completed = False
                    break
                pbar.set_postfix(loss=result["loss"])

        if completed and result is not None and self._log_interval_due("epochs"):
            self.log(result, step=self.training_state.current_epoch)
        if self._log_interval_due("epochs"):
            self._sync_halt()
        if completed:
            self.training_state.current_step_in_epoch = 0
        self.trigger_callbacks(event=CallbackEvent.EPOCH_END)
        return completed

    def train_a_step(self, batch: Dict[str, Any]) -> dict:
        """Run forward, backward, optional clip/step, and logging for one batch.

        Args:
            batch: Training batch moved to ``training_config.device``.
        """
        self.trigger_callbacks(event=CallbackEvent.STEP_START, batch=batch)
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        device = self.training_config.device
        model.train()
        for key, value in batch.items():
            if type(value) is torch.Tensor:
                batch[key] = value.to(device)
        model = model.to(device)

        raw_loss = self.compute_loss(model, batch)
        if isinstance(raw_loss, Tensor):
            loss = raw_loss
        else:
            loss = cast(Tensor, cast(Dict[str, Any], raw_loss)["loss"])
        loss = loss / self.training_config.gradient_accumulation_steps
        self.trigger_callbacks(event=CallbackEvent.PRE_BACKWARD, loss=loss)
        loss.backward()
        self.trigger_callbacks(event=CallbackEvent.POST_BACKWARD, loss=loss)

        if ((self.training_state.current_global_step + 1) % self.training_config.gradient_accumulation_steps) == 0:
            self.gradient_clip()
            self.trigger_callbacks(event=CallbackEvent.PRE_OPTIMIZER_STEP)
            optimizer.step()
            self.trigger_callbacks(event=CallbackEvent.POST_OPTIMIZER_STEP)
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        result = {"loss": loss.item()}
        self._include_model_metrics(result, raw_loss)
        if self._log_interval_due("steps"):
            self.log(result, step=self.training_state.current_global_step)
        self.trigger_callbacks(event=CallbackEvent.STEP_END, batch=batch)
        return result

    def gradient_clip(self) -> None:
        """Clip gradients when ``grad_clip_strategy`` is ``norm`` or ``value``."""
        model = self.model
        if self.training_config.grad_clip_strategy == "norm":
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=self.training_config.max_grad_norm,
                norm_type=2,
                error_if_nonfinite=False,
            )
        if self.training_config.grad_clip_strategy == "value":
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                clip_value=self.training_config.max_grad_value,
            )

    def eval(self):
        """Evaluate the model; subclasses implement distributed gathering."""

    def _halt_if_any_rank_failed(self) -> bool:
        """Return whether a rank-local failure should stop this training loop.

        A single-process pipeline stops when this process remembered a failure.
        DistributedPipeline shares that fact across ranks before returning.
        """
        return has_deferred_error()

    def _sync_halt(self) -> None:
        """Stop this rank when a shared or local failure is waiting."""
        if self._halt_if_any_rank_failed():
            self._halt = True

    @property
    def halted(self) -> bool:
        """Whether a shared or local failure stopped training on this rank."""
        return self._halt

    def _is_global_main_process(self) -> bool:
        """Return whether this process records metrics.

        A pipeline without an accelerator is the only process.
        """
        accelerator = getattr(self, "accelerator", None)
        return True if accelerator is None else bool(accelerator.is_main_process)

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
        """Record one metric payload on the global main process.

        Callbacks receive a ``LOG`` event. A callback failure is remembered, and
        the text line is still written when a logger is attached.

        Args:
            metrics: values to record.
            step: step index for this payload.
        """
        if not self._is_global_main_process():
            return
        try:
            self.trigger_callbacks(event=CallbackEvent.LOG, metrics=metrics, step=step)
        except Exception as exc:
            defer_error(exc)
            if self.logger is not None:
                self.logger.error("log failed: %s", exc)
        if self.logger is not None:
            self.logger.info(_format_log(metrics, step))

    def _include_model_metrics(self, result: Dict[str, Any], raw_loss: object) -> None:
        """Copy ``metrics`` from a model output into the step result.

        Args:
            result: payload that will be logged for this step.
            raw_loss: value returned by ``compute_loss``.
        """
        if not isinstance(raw_loss, dict):
            return
        payload = cast(dict[str, Any], raw_loss)
        model_output = payload.get("model_output")
        source = cast(dict[str, Any], model_output) if isinstance(model_output, dict) else payload
        extra = source.get("metrics")
        if isinstance(extra, dict):
            result.update(cast(dict[str, Any], extra))

    def _is_local_main_process(self) -> bool:
        """Return True if this process should emit UI output (e.g. progress bars).

        Plain BasePipeline has no accelerator and is always the local main process.
        DistributedPipeline subclasses expose ``self.accelerator``.
        """
        accelerator = getattr(self, "accelerator", None)
        return True if accelerator is None else accelerator.is_local_main_process

    def _cleanup_old_checkpoints(self, save_dir):
        checkpoints = [d for d in os.listdir(save_dir) if d.startswith("checkpoint_")]

        def _checkpoint_sort_key(name: str) -> int:
            m = re.search(r"global(\d+)", name)
            return int(m.group(1)) if m is not None else 0

        checkpoints.sort(key=_checkpoint_sort_key)

        while len(checkpoints) > self.training_config.save_total_limit:
            oldest_checkpoint = checkpoints.pop(0)
            oldest_path = os.path.join(save_dir, oldest_checkpoint)
            shutil.rmtree(oldest_path)
            if self.logger is not None:
                self.logger.info(f"Deleted old checkpoint: {oldest_path}")

    def _get_checkpoint_name(self, epoch, step, global_step):
        return f"checkpoint_epoch{epoch}_step{step}_global{global_step}"

    def _can_save(self, flag: Literal["epochs", "steps"]):
        if (
            self.training_config.save_strategy != flag
            or self.training_config.save_dir is None
            or self.training_config.save_dir == ""
        ):
            return False
        if flag == "epochs":
            # current_epoch is already incremented after train_an_epoch() completes.
            return (
                self.training_state.current_epoch > 0
                and self.training_state.current_epoch % self.training_config.save_epochs == 0
            )
        if flag == "steps":
            step = self.training_state.current_global_step
            return step > 0 and step % self.training_config.save_steps == 0
        return False

    def _log_interval_due(self, flag: Literal["epochs", "steps"]) -> bool:
        """Return whether this step or epoch is a logging point on every rank.

        The logger is not consulted. Ranks agree because they share the log
        strategy and the training counters.

        Args:
            flag: ``"epochs"`` or ``"steps"``.

        Returns:
            True when the current counter matches the logging interval.
        """
        if self.log_config.log_strategy != flag or self.log_config.log_dir is None or self.log_config.log_dir == "":
            return False
        if flag == "epochs":
            return (self.training_state.current_epoch + 1) % self.log_config.log_epochs == 0
        if flag == "steps":
            return (self.training_state.current_global_step + 1) % self.log_config.log_steps == 0
        return False

    def _can_eval(self, flag: Literal["epochs", "steps"]):
        if self.training_config.eval_strategy != flag:
            return False
        if flag == "epochs":
            # current_epoch is already incremented after train_an_epoch() completes.
            return (
                self.training_state.current_epoch > 0
                and self.training_state.current_epoch % self.training_config.eval_epochs == 0
            )
        if flag == "steps":
            step = self.training_state.current_global_step
            return step > 0 and step % self.training_config.eval_steps == 0
        return False

    def save(self) -> None:
        """Write model, optimizer, and training state under ``save_dir``."""
        self.trigger_callbacks(event=CallbackEvent.PRE_SAVE)
        save_dir = self.training_config.save_dir
        if save_dir is None or save_dir == "":
            return
        os.makedirs(save_dir, exist_ok=True)

        epoch = self.training_state.current_epoch
        step = self.training_state.current_step_in_epoch
        global_step = self.training_state.current_global_step
        checkpoint_name = self._get_checkpoint_name(epoch, step, global_step)
        temp_checkpoint_dir = os.path.join(save_dir, f"tmp_{checkpoint_name}")
        final_checkpoint_dir = os.path.join(save_dir, checkpoint_name)
        if os.path.exists(final_checkpoint_dir):
            return

        os.makedirs(temp_checkpoint_dir, exist_ok=True)
        try:
            for base_name in [
                "model",
                "optimizer",
                "scheduler",
                "training_state",
                "training_config",
                "log_config",
            ]:
                file_path = os.path.join(temp_checkpoint_dir, f"{base_name}.pth")
                torch.save(getattr(self, base_name), file_path)
            self.trigger_callbacks(event=CallbackEvent.PRE_FINALIZE_SAVE, checkpoint_dir=temp_checkpoint_dir)
            os.rename(temp_checkpoint_dir, final_checkpoint_dir)
            if self.logger is not None:
                self.logger.info(f"Model saved to {final_checkpoint_dir}")
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to save checkpoint: {e}")
            shutil.rmtree(temp_checkpoint_dir, ignore_errors=True)
            raise
        try:
            self._cleanup_old_checkpoints(save_dir=save_dir)
        except Exception as exc:
            if self.logger is not None:
                self.logger.error(f"Failed to delete old checkpoints: {exc}")
        self.trigger_callbacks(event=CallbackEvent.POST_SAVE)

    def load(self) -> None:
        """Restore the latest checkpoint from ``save_dir`` when present."""
        self.trigger_callbacks(event=CallbackEvent.PRE_LOAD)
        # check load condition ============================================================================
        checkpoint_dir = self.get_latest_checkpoint_dir()
        if checkpoint_dir is None or len(os.listdir(checkpoint_dir)) <= 0:
            return

        # load ============================================================================================

        for base_name in [
            "model",
            "optimizer",
            "scheduler",
            "training_state",
            "training_config",
            "log_config",
        ]:
            file_path = os.path.join(checkpoint_dir, f"{base_name}.pth")
            if not os.path.exists(file_path):
                if self.logger is not None:
                    self.logger.info(f"File {file_path} does not exist")
                continue
            setattr(self, base_name, torch.load(file_path, weights_only=False))

        if self.logger is not None:
            self.logger.info(f"Model loaded from {checkpoint_dir}")
        self.trigger_callbacks(event=CallbackEvent.POST_LOAD)

    def get_latest_checkpoint_dir(self) -> str | None:
        return latest_checkpoint_dir(self.training_config.save_dir)

    def add_callbacks(self, callbacks: List[BaseCallback] | None):
        """Register extra pipeline callbacks.

        Args:
            callbacks: Callbacks to append, or ``None`` to skip.
        """
        self.callback_manager.add_callbacks(callbacks=callbacks)

    def trigger_callbacks(self, event: CallbackEvent, **kwargs):
        """Dispatch one callback event with training context and extra keywords.

        Args:
            event: Callback lifecycle event.
            **kwargs: Event-specific payload for callbacks.
        """
        self.callback_manager.trigger(
            event=event,
            training_config=self.training_config,
            log_config=self.log_config,
            training_state=self.training_state,
            logger=self.logger,
            model=self.model,
            dataloader=self.dataloader,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            **kwargs,
        )
