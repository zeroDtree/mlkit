"""run_experiment drives any Task through one process-group lifecycle.

Run with:
    uv run pytest tests/pipeline/test_run_task.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from mlkit.util.experiment import RunContext, Task, run_experiment


def _cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "job": {"seed": 7, "mixed_precision": "no"},
            "log": {"log_dir": "logs"},
            "reporter": "none",
            "mlflow": {},
            "wandb": {},
        }
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    is_main: bool,
    num_processes: int = 1,
) -> list[dict[str, Any]]:
    accelerator = MagicMock()
    accelerator.is_main_process = is_main
    accelerator.num_processes = num_processes
    accelerator.device = torch.device("cpu")
    accelerator.check_trigger.return_value = False
    accelerator.wait_for_everyone.side_effect = lambda: events.append("wait")
    accelerator.end_training.side_effect = lambda: events.append("end")
    built: list[dict[str, Any]] = []

    def build(*args: Any, **kwargs: Any) -> MagicMock:
        del args
        built.append(kwargs)
        return accelerator

    monkeypatch.setattr("accelerate.Accelerator", build)
    monkeypatch.setattr("mlkit.util.seed.seed_everything", lambda seed: events.append(f"seed={seed}"))
    monkeypatch.setattr("mlkit.util.log.get_logger", lambda *_args, **_kwargs: logging.getLogger("mlkit.test.task"))
    monkeypatch.setattr("mlkit.util.log.get_and_create_new_log_dir", lambda *_args, **_kwargs: "logs")
    return built


@dataclass(kw_only=True)
class _Run(RunContext):
    value: int


class _Job(Task[_Run]):
    section: ClassVar[str] = "job"

    def __init__(self, events: list[str], error: BaseException | None = None) -> None:
        self.events = events
        self.error = error

    def run_name(self, cfg: DictConfig) -> str:
        del cfg
        return "job"

    def run(self, ctx: RunContext) -> _Run:
        self.events.append("run")
        if self.error is not None:
            raise self.error
        return _Run(cfg=ctx.cfg, accelerator=ctx.accelerator, logger=ctx.logger, reporter=ctx.reporter, value=42)

    def finish(self, run: _Run) -> None:
        self.events.append(f"finish={run.value}")


def test_the_task_section_supplies_seed_and_mixed_precision(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    built = _install(monkeypatch, events, is_main=True)

    run_experiment(_cfg(), _Job(events))

    assert events[0] == "seed=7"
    assert built == [{"mixed_precision": "no"}]


def test_finish_runs_on_main_after_teardown_and_the_run_is_returned(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events, is_main=True)

    run = run_experiment(_cfg(), _Job(events))

    assert events == ["seed=7", "run", "wait", "end", "finish=42"]
    assert run.value == 42
    assert run.logger is not None
    assert run.reporter is not None


def test_non_main_returns_the_run_without_finish(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events, is_main=False, num_processes=2)

    run = run_experiment(_cfg(), _Job(events))

    assert events == ["seed=7", "run", "wait", "end"]
    assert run.value == 42
    assert run.logger is None
    assert run.reporter is None


def test_run_error_ends_the_group_without_finish(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events, is_main=True)

    with pytest.raises(RuntimeError, match="boom"):
        run_experiment(_cfg(), _Job(events, error=RuntimeError("boom")))

    assert events == ["seed=7", "run", "end"]
