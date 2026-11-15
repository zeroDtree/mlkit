"""Pipeline.log writes one text line and remembers the first callback failure.

Run with:
    uv run pytest tests/pipeline/test_pipeline_log.py
"""

from __future__ import annotations

import logging

import pytest

from mlkit.pipeline.callback import BaseCallback, CallbackEvent, CallbackManager
from mlkit.pipeline.pipeline import BasePipeline
from mlkit.util.failure import clear_deferred_error, has_deferred_error, raise_deferred_error


class RecordingLog(BaseCallback):
    def __init__(self) -> None:
        self.calls: list[tuple[dict, int | None]] = []

    def on_event(self, event: CallbackEvent, *args: object, **kwargs: object) -> None:
        del args
        if event is not CallbackEvent.LOG:
            return
        metrics = kwargs["metrics"]
        step = kwargs["step"]
        assert isinstance(metrics, dict)
        assert step is None or isinstance(step, int)
        self.calls.append((metrics, step))


class BoomLog(BaseCallback):
    def __init__(self) -> None:
        self.n = 0

    def on_event(self, event: CallbackEvent, *args: object, **kwargs: object) -> None:
        del args, kwargs
        if event is not CallbackEvent.LOG:
            return
        self.n += 1
        raise RuntimeError(f"backend {self.n}")


class _Pipeline(BasePipeline):
    def compute_loss(self, model: object, batch: object) -> dict[str, float]:
        del model, batch
        return {"loss": 0.0}


def _pipeline(logger: logging.Logger | None, callbacks: list[BaseCallback], *, main: bool = True) -> _Pipeline:
    pipeline = object.__new__(_Pipeline)
    pipeline.callback_manager = CallbackManager()
    pipeline.callback_manager.add_callbacks(callbacks)
    pipeline.logger = logger
    pipeline.accelerator = None if main else type("Accelerator", (), {"is_main_process": False})()
    pipeline.training_config = None
    pipeline.log_config = None
    pipeline.training_state = None
    pipeline.model = None
    pipeline.dataloader = None
    pipeline.optimizer = None
    pipeline.scheduler = None
    return pipeline


def test_log_writes_the_text_line_and_notifies_callbacks(caplog: pytest.LogCaptureFixture):
    logger = logging.getLogger("mlkit.test.pipeline.log.ok")
    logger.setLevel(logging.INFO)
    recorder = RecordingLog()
    pipeline = _pipeline(logger, [recorder])
    with caplog.at_level(logging.INFO, logger=logger.name):
        pipeline.log({"loss": 1.5, "note": "x"}, step=3)

    assert recorder.calls == [({"loss": 1.5, "note": "x"}, 3)]
    messages = [record.message for record in caplog.records if record.name == logger.name]
    assert messages == ["step=3 loss=1.5 note=x"]


def test_log_without_a_logger_still_notifies_callbacks(caplog: pytest.LogCaptureFixture):
    recorder = RecordingLog()
    pipeline = _pipeline(None, [recorder])
    with caplog.at_level(logging.INFO):
        pipeline.log({"loss": 1.0}, step=1)

    assert recorder.calls == [({"loss": 1.0}, 1)]
    assert caplog.records == []


def test_log_keeps_the_first_callback_error_and_writes_text(caplog: pytest.LogCaptureFixture):
    logger = logging.getLogger("mlkit.test.pipeline.log.defer")
    logger.setLevel(logging.INFO)
    pipeline = _pipeline(logger, [BoomLog()])
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            pipeline.log({"loss": 1.0}, step=3)
            pipeline.log({"loss": 2.0}, step=4)
        assert has_deferred_error()
        messages = [record.message for record in caplog.records if record.name == logger.name]
        assert "step=3 loss=1.0" in messages
        assert "step=4 loss=2.0" in messages
        assert any(message == "log failed: backend 1" for message in messages)
        with pytest.raises(RuntimeError, match="backend 1"):
            raise_deferred_error()
        assert not has_deferred_error()
    finally:
        clear_deferred_error()


def test_log_on_a_non_main_process_is_silent(caplog: pytest.LogCaptureFixture):
    logger = logging.getLogger("mlkit.test.pipeline.log.other")
    logger.setLevel(logging.INFO)
    recorder = RecordingLog()
    pipeline = _pipeline(logger, [recorder], main=False)
    with caplog.at_level(logging.INFO, logger=logger.name):
        pipeline.log({"loss": 1.0}, step=1)
    assert recorder.calls == []
    assert caplog.records == []


def test_model_metrics_are_merged_from_the_model_output():
    pipeline = object.__new__(_Pipeline)
    result = {"loss": 1.0}
    raw_loss = {"loss": 1.0, "model_output": {"metrics": {"train/loss": 0.5}}}
    pipeline._include_model_metrics(result, raw_loss)
    assert result == {"loss": 1.0, "train/loss": 0.5}

    plain = {"loss": 1.0}
    pipeline._include_model_metrics(plain, {"metrics": {"train/kl_loss": 0.25}})
    assert plain["train/kl_loss"] == 0.25

    untouched = {"loss": 1.0}
    pipeline._include_model_metrics(untouched, {"metrics": {"nope": 1}, "model_output": {}})
    assert untouched == {"loss": 1.0}
