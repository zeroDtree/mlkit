"""run_experiment destroys the process group before post-training work.

Run with:
    uv run pytest tests/pipeline/test_run_experiment_shutdown.py
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from mlkit.util.experiment import ConfigSnapshotCallback, TrainSpec, run_experiment
from mlkit.util.failure import defer_error, has_deferred_error


def _cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "train": {
                "seed": 0,
                "mixed_precision": "no",
                "train_strategy": "steps",
                "n_epochs": 1,
                "n_steps": 1,
                "batch_size": 2,
                "device": "cpu",
                "save_strategy": None,
                "save_dir": "checkpoints",
                "save_steps": 10,
                "save_epochs": 1,
                "save_total_limit": 1,
                "num_workers": 0,
                "train_shuffle": False,
                "eval_strategy": None,
                "eval_steps": 1,
                "eval_epochs": 1,
                "grad_clip_strategy": None,
                "max_grad_norm": 1.0,
                "max_grad_value": 1.0,
                "gradient_accumulation_steps": 1,
                "n_warmup_steps": 0,
                "lr_scheduler_type": "constant",
            },
            "log": {
                "log_dir": "logs",
                "log_steps": 1,
                "log_epochs": 1,
                "log_strategy": "steps",
            },
            "reporter": "none",
            "mlflow": {},
            "wandb": {},
            "dataset": {"name": "toy"},
            "model": {"name": "toy"},
            "optimizer": {"name": "SGD", "lr": 0.01},
        }
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    is_main: bool,
    num_processes: int = 1,
    train_error: BaseException | None = None,
    on_train: Callable[[], None] | None = None,
    configs: list[Any] | None = None,
    pipelines: list[dict[str, Any]] | None = None,
    halt: bool = False,
) -> MagicMock:
    accelerator = MagicMock()
    accelerator.is_local_main_process = is_main
    accelerator.is_main_process = is_main
    accelerator.device = torch.device("cpu")
    accelerator.num_processes = num_processes
    accelerator.check_trigger.return_value = False
    accelerator.wait_for_everyone.side_effect = lambda: events.append("wait")
    accelerator.end_training.side_effect = lambda: events.append("end")
    monkeypatch.setattr("accelerate.Accelerator", lambda *_args, **_kwargs: accelerator)

    class Pipeline:
        def __init__(self, **kwargs: Any) -> None:
            if pipelines is not None:
                pipelines.append(kwargs)
            self.halted = False

        def train(self) -> None:
            events.append("train")
            if on_train is not None:
                on_train()
            if train_error is not None:
                raise train_error
            # The real pipeline sets this after sharing a deferred failure.
            if halt or has_deferred_error():
                self.halted = True

    class TrainingConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)
            if configs is not None:
                configs.append(self)

    monkeypatch.setattr("mlkit.util.utils_for_main.get_train_class", lambda: (Pipeline, TrainingConfig))
    monkeypatch.setattr("mlkit.util.utils_for_main.get_learing_rate_scheduler", lambda *_args, **_kwargs: MagicMock())
    monkeypatch.setattr(
        "mlkit.util.log.get_logger",
        lambda *_args, **_kwargs: logging.getLogger("mlkit.test.run_experiment_shutdown"),
    )
    monkeypatch.setattr("mlkit.util.log.get_and_create_new_log_dir", lambda *_args, **_kwargs: "logs")
    return accelerator


def _run(
    _events: list[str],
    *,
    on_finish: Callable | None = None,
    after_train: Callable | None = None,
    bundle: dict[str, Any] | None = None,
    run_suffix: str = "",
) -> None:
    model = torch.nn.Linear(1, 1)

    def get_model(cfg: DictConfig, **kwargs: Any):
        del cfg, kwargs
        return {"model": model, **(bundle or {})}

    def get_dataset(cfg: DictConfig):
        del cfg
        return [0, 1], None, None

    def get_collate(cfg: DictConfig):
        del cfg
        return None

    spec = TrainSpec(
        get_model=get_model,
        get_dataset=get_dataset,
        get_collate=get_collate,
        on_finish=on_finish,
        after_train=after_train,
        run_suffix=run_suffix,
    )
    run_experiment(_cfg(), spec)


class _ReporterStub:
    def __init__(self, callback: object) -> None:
        self.callback = callback

    def store_config(self, config: object) -> None:
        del config

    def as_callback(self) -> object:
        return self.callback


def _install_reporter(monkeypatch: pytest.MonkeyPatch, callback: object, run_names: list[str]) -> None:
    class _Scope:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            run_names.append(cast(str, kwargs["run_name"]))

        def __enter__(self) -> _ReporterStub:
            return _ReporterStub(callback)

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr("mlkit.util.report.ReporterContext", _Scope)


def test_bundle_callbacks_and_seed_reach_the_pipeline_constructor(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    pipelines: list[dict[str, Any]] = []
    _install(monkeypatch, events, is_main=True, pipelines=pipelines)
    bundle_callback, reporter_callback = object(), object()
    _install_reporter(monkeypatch, reporter_callback, [])

    _run(events, bundle={"callbacks": [bundle_callback]})

    bundle_cb, snapshot_cb, last_cb = pipelines[0]["callbacks"]
    assert (bundle_cb, last_cb) == (bundle_callback, reporter_callback)
    assert isinstance(snapshot_cb, ConfigSnapshotCallback)
    assert pipelines[0]["seed"] == 0


def test_run_suffix_names_the_run_and_the_checkpoint_dir(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    configs: list[Any] = []
    run_names: list[str] = []
    log_dirs: list[tuple[str, dict[str, Any]]] = []
    _install(monkeypatch, events, is_main=True, configs=configs)
    _install_reporter(monkeypatch, object(), run_names)

    def create_log_dir(root: str, **kwargs: Any) -> str:
        log_dirs.append((root, kwargs))
        return "logs"

    monkeypatch.setattr("mlkit.util.log.get_and_create_new_log_dir", create_log_dir)

    _run(events, run_suffix="-sweep:1")

    assert run_names == ["toy-toy-lr:0.01-n_steps:1-sweep:1"]
    assert [config.save_dir for config in configs] == ["checkpoints/toy/toy/-lr:0.01-bs:2-sweep:1"]
    assert log_dirs == [("logs", {"suffix": str(os.getpid())})]


def test_empty_save_dir_disables_checkpointing(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    configs: list[Any] = []
    _install(monkeypatch, events, is_main=True, configs=configs)
    cfg = _cfg()
    cfg.train.save_dir = ""

    def get_model(cfg: DictConfig, **kwargs: Any):
        del cfg, kwargs
        return {"model": torch.nn.Linear(1, 1)}

    def get_dataset(cfg: DictConfig):
        del cfg
        return [0, 1], None, None

    def get_collate(cfg: DictConfig):
        del cfg
        return None

    spec = TrainSpec(get_model=get_model, get_dataset=get_dataset, get_collate=get_collate)
    run_experiment(cfg, spec)

    assert [config.save_dir for config in configs] == [""]


def test_every_rank_uses_the_same_checkpoint_dir(monkeypatch: pytest.MonkeyPatch):
    expected = "checkpoints/toy/toy/-lr:0.01-bs:2"

    def on_finish(ctx) -> None:
        del ctx

    for is_main in (True, False):
        events: list[str] = []
        configs: list[Any] = []
        _install(monkeypatch, events, is_main=is_main, configs=configs)
        _run(events, on_finish=on_finish)
        assert [config.save_dir for config in configs] == [expected]


def test_main_process_destroys_group_before_post_train(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True)

    def on_finish(ctx) -> None:
        del ctx.cfg, ctx.model_bundle, ctx.pipeline, ctx.train_set
        assert ctx.accelerator is accelerator
        events.append("post")

    _run(events, on_finish=on_finish)
    assert events == ["train", "wait", "end", "post"]


def test_non_main_process_destroys_group_without_post_train(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events, is_main=False)

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    _run(events, on_finish=on_finish)
    assert events == ["train", "wait", "end"]


def test_train_error_destroys_group_without_barrier(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True, train_error=RuntimeError("boom"))

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    with pytest.raises(RuntimeError, match="boom"):
        _run(events, on_finish=on_finish)

    assert events == ["train", "end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_end_training_error_keeps_the_train_error(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True, train_error=RuntimeError("boom"))

    def end_training() -> None:
        events.append("end")
        raise RuntimeError("end failed")

    accelerator.end_training.side_effect = end_training

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    with pytest.raises(RuntimeError, match="boom"):
        _run(events, on_finish=on_finish)

    assert events == ["train", "end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_post_train_error_stays_inside_reporter(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    seen: list[BaseException] = []
    _install(monkeypatch, events, is_main=True)

    class _Open:
        def store_config(self, config: object) -> None:
            del config

        def as_callback(self) -> "_Open":
            return self

    class _OpenScope:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> _Open:
            return _Open()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, tb
            if isinstance(exc, BaseException):
                seen.append(exc)

    monkeypatch.setattr("mlkit.util.report.ReporterContext", _OpenScope)

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")
        raise RuntimeError("post failed")

    with pytest.raises(RuntimeError, match="post failed"):
        _run(events, on_finish=on_finish)

    assert events == ["train", "wait", "end", "post"]
    assert [type(exc) for exc in seen] == [RuntimeError]


def test_deferred_error_is_raised_after_teardown(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(
        monkeypatch,
        events,
        is_main=True,
        on_train=lambda: defer_error(RuntimeError("log failed")),
    )

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    with pytest.raises(RuntimeError, match="log failed"):
        _run(events, on_finish=on_finish)

    assert events == ["train", "wait", "end"]


def test_deferred_error_skips_after_train_on_main(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(
        monkeypatch,
        events,
        is_main=True,
        on_train=lambda: defer_error(RuntimeError("log failed")),
    )

    def after_train(ctx) -> None:
        del ctx
        events.append("after")

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    with pytest.raises(RuntimeError, match="log failed"):
        _run(events, on_finish=on_finish, after_train=after_train)

    assert events == ["train", "wait", "end"]


def test_shared_halt_skips_after_train_on_non_main(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=False, halt=True)

    def after_train(ctx) -> None:
        del ctx
        events.append("after")

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    _run(events, on_finish=on_finish, after_train=after_train)

    assert events == ["train", "wait", "end"]
    accelerator.set_trigger.assert_not_called()


def test_scheduler_rewrite_replaces_the_stored_config(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    opened: list[dict[str, Any]] = []
    stored: list[dict[str, Any]] = []
    _install(monkeypatch, events, is_main=True)

    def rewrite_scheduler(optimizer, accelerator, train_set, cfg):
        del optimizer, accelerator, train_set
        cfg.train.n_steps = 99
        return MagicMock()

    monkeypatch.setattr("mlkit.util.utils_for_main.get_learing_rate_scheduler", rewrite_scheduler)

    class _Recording:
        def store_config(self, config: object) -> None:
            assert isinstance(config, dict)
            stored.append(cast(dict[str, Any], config))

        def as_callback(self) -> "_Recording":
            return self

    class _RecordingScope:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            config = kwargs["config"]
            assert isinstance(config, dict)
            opened.append(cast(dict[str, Any], config))

        def __enter__(self) -> _Recording:
            return _Recording()

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb

    monkeypatch.setattr("mlkit.util.report.ReporterContext", _RecordingScope)

    def on_finish(ctx) -> None:
        del ctx

    _run(events, on_finish=on_finish)

    assert opened[0]["train"]["n_steps"] == 1
    assert stored[-1]["train"]["n_steps"] == 99


def test_after_train_runs_on_main_before_teardown(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events, is_main=True)

    def after_train(ctx) -> None:
        del ctx
        events.append("after")

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    _run(events, on_finish=on_finish, after_train=after_train)
    assert events == ["train", "after", "wait", "end", "post"]


def test_after_train_runs_on_non_main_without_post_train(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events, is_main=False)

    def after_train(ctx) -> None:
        del ctx
        events.append("after")

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    _run(events, on_finish=on_finish, after_train=after_train)
    assert events == ["train", "after", "wait", "end"]


def test_after_train_error_destroys_group_without_barrier(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True)

    def after_train(ctx) -> None:
        del ctx
        events.append("after")
        raise RuntimeError("after failed")

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    with pytest.raises(RuntimeError, match="after failed"):
        _run(events, on_finish=on_finish, after_train=after_train)

    assert events == ["train", "after", "end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_log_dir_error_destroys_group_without_training(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True)

    def create_log_dir(*args, **kwargs):
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr("mlkit.util.log.get_and_create_new_log_dir", create_log_dir)

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    with pytest.raises(OSError, match="disk full"):
        _run(events, on_finish=on_finish)

    assert events == ["end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_reporter_open_error_destroys_group_without_training(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True)

    class _FailOpenScope:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> object:
            raise RuntimeError("mlflow down")

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr("mlkit.util.report.ReporterContext", _FailOpenScope)

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    with pytest.raises(RuntimeError, match="mlflow down"):
        _run(events, on_finish=on_finish)

    assert events == ["end"]
    accelerator.wait_for_everyone.assert_not_called()
    accelerator.set_trigger.assert_not_called()


def test_end_training_error_keeps_the_reporter_open_error(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True)

    def end_training() -> None:
        events.append("end")
        raise RuntimeError("end failed")

    accelerator.end_training.side_effect = end_training

    class _FailOpenScope:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> object:
            raise RuntimeError("mlflow down")

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr("mlkit.util.report.ReporterContext", _FailOpenScope)

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    with pytest.raises(RuntimeError, match="mlflow down"):
        _run(events, on_finish=on_finish)

    assert events == ["end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_reporter_open_is_shared_before_training(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True, num_processes=2)
    accelerator.check_trigger.side_effect = lambda: events.append("sync") or False

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    _run(events, on_finish=on_finish)
    assert events == ["sync", "train", "wait", "end", "post"]
    accelerator.set_trigger.assert_not_called()


def test_non_main_waits_for_reporter_open_before_training(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=False, num_processes=2)
    accelerator.check_trigger.side_effect = lambda: events.append("sync") or False

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    _run(events, on_finish=on_finish)
    assert events == ["sync", "train", "wait", "end"]
    accelerator.set_trigger.assert_not_called()


def test_reporter_open_error_sets_the_trigger(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True, num_processes=2)
    accelerator.set_trigger.side_effect = lambda: events.append("trigger")
    accelerator.check_trigger.side_effect = lambda: events.append("sync") or True

    class _FailOpenScope:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> object:
            raise RuntimeError("mlflow down")

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr("mlkit.util.report.ReporterContext", _FailOpenScope)

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    with pytest.raises(RuntimeError, match="mlflow down"):
        _run(events, on_finish=on_finish)

    assert events == ["trigger", "sync", "end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_other_rank_stops_when_reporter_open_fails(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=False, num_processes=2)
    accelerator.check_trigger.side_effect = lambda: events.append("sync") or True

    class _AssertScope:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("open")
            raise AssertionError("non-main rank opened the reporter")

        def __enter__(self) -> object:
            raise AssertionError("unreachable")

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr("mlkit.util.report.ReporterContext", _AssertScope)

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    with pytest.raises(RuntimeError, match="Reporter failed to open on another rank"):
        _run(events, on_finish=on_finish)

    assert events == ["sync", "end"]
    accelerator.set_trigger.assert_not_called()
    accelerator.wait_for_everyone.assert_not_called()


def test_local_main_that_is_not_global_main_skips_the_reporter(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=False, num_processes=2)
    accelerator.is_local_main_process = True
    accelerator.is_main_process = False
    accelerator.check_trigger.side_effect = lambda: events.append("sync") or False

    class _AssertScope:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("open")
            raise AssertionError("local main opened the reporter")

        def __enter__(self) -> object:
            raise AssertionError("unreachable")

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr("mlkit.util.report.ReporterContext", _AssertScope)

    def on_finish(ctx) -> None:
        del ctx
        events.append("post")

    _run(events, on_finish=on_finish)
    assert events == ["sync", "train", "wait", "end"]
    accelerator.set_trigger.assert_not_called()
