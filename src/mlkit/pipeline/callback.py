from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List, Optional


class CallbackEvent(Enum):
    # training
    TRAINING_START = "training_start"
    TRAINING_END = "training_end"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    LOG = "log"
    # save, load
    PRE_SAVE = "pre_save"
    PRE_FINALIZE_SAVE = "pre_finalize_save"
    POST_SAVE = "post_save"
    PRE_LOAD = "pre_load"
    POST_LOAD = "post_load"
    # optimize
    PRE_COMPUTE_LOSS = "pre_compute_loss"
    POST_COMPUTE_LOSS = "post_compute_loss"
    PRE_BACKWARD = "pre_backward"
    POST_BACKWARD = "post_backward"
    PRE_OPTIMIZER_STEP = "pre_optimizer_step"
    POST_OPTIMIZER_STEP = "post_optimizer_step"
    # eval
    PRE_EVAL = "pre_eval"
    POST_EVAL = "post_eval"
    PRE_EVAL_STEP = "pre_eval_step"
    POST_EVAL_STEP = "post_eval_step"


class BaseCallback(metaclass=ABCMeta):
    @abstractmethod
    def on_event(self, event: CallbackEvent, *args, **kwargs):
        """Handle one pipeline lifecycle event.

        Args:
            event: Callback event.
            *args: Positional payload for the event.
            **kwargs: Keyword payload for the event.
        """


class CallbackManager:
    def __init__(self):
        self.callbacks: List[BaseCallback] = []

    def add_callback(self, callback: BaseCallback):
        """Append one callback.

        Args:
            callback: Callback instance to register.
        """
        if callback is not None:
            self.callbacks.append(callback)

    def add_callbacks(self, callbacks: Optional[List[BaseCallback]]):
        """Append multiple callbacks.

        Args:
            callbacks: Callbacks to register.
        """
        if callbacks is not None and len(callbacks) > 0:
            self.callbacks.extend(callbacks)

    def trigger(self, event: CallbackEvent, *args, **kwargs):
        """Invoke every registered callback for ``event``.

        Args:
            event: Callback event.
            *args: Positional payload.
            **kwargs: Keyword payload.
        """
        for callback in self.callbacks:
            callback.on_event(event, *args, **kwargs)
