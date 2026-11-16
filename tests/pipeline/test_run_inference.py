"""InferSpec loads trained weights, then runs inference hooks around teardown.

Run with:
    uv run pytest tests/pipeline/test_run_inference.py
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from mlkit.util.experiment import InferRun, InferSpec, run_experiment
from mlkit.util.experiment.runner import GetDatasetFn, GetModelFn


def _cfg(tmp_path: Path, checkpoint_dir: str | None = None) -> DictConfig:
    return OmegaConf.create(
        {
            "train": {
                "save_dir": str(tmp_path / "checkpoints"),
                "batch_size": 2,
                "train_strategy": "steps",
                "n_steps": 1,
            },
            "infer": {"seed": 0, "mixed_precision": "no", "checkpoint_dir": checkpoint_dir},
            "log": {"log_dir": "logs"},
            "reporter": "none",
            "mlflow": {},
            "wandb": {},
            "dataset": {"name": "toy"},
            "model": {"name": "toy"},
            "optimizer": {"name": "SGD", "lr": 0.01},
        }
    )


def _save_dir(tmp_path: Path, suffix: str = "") -> Path:
    return Path(f"{tmp_path / 'checkpoints' / 'toy' / 'toy' / '-lr:0.01-bs:2'}{suffix}")


def _write_checkpoint(directory: Path, weight: float) -> Path:
    from safetensors.torch import save_model

    model = torch.nn.Linear(1, 1)
    torch.nn.init.constant_(model.weight, weight)
    directory.mkdir(parents=True)
    save_model(model, str(directory / "model.safetensors"))
    return directory


def _install(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> MagicMock:
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.num_processes = 1
    accelerator.device = torch.device("cpu")
    accelerator.wait_for_everyone.side_effect = lambda: events.append("wait")
    accelerator.end_training.side_effect = lambda: events.append("end")
    monkeypatch.setattr("accelerate.Accelerator", lambda *_args, **_kwargs: accelerator)
    monkeypatch.setattr("mlkit.util.log.get_logger", lambda *_args, **_kwargs: logging.getLogger("mlkit.test.infer"))
    monkeypatch.setattr("mlkit.util.log.get_and_create_new_log_dir", lambda *_args, **_kwargs: "logs")
    return accelerator


def _spec(
    events: list[str],
    *,
    bundle: dict[str, Any] | None = None,
    get_model: GetModelFn | None = None,
    get_dataset: GetDatasetFn | None = None,
    infer: Callable[[InferRun], Any] | None = None,
    run_suffix: str = "",
) -> InferSpec:
    def default_get_model(cfg: DictConfig) -> dict[str, Any]:
        del cfg
        return {"model": torch.nn.Linear(1, 1), **(bundle or {})}

    def post_dist_teardown(run: InferRun) -> None:
        events.append(f"post={run.outputs}")

    return InferSpec(
        get_model=get_model or default_get_model,
        get_dataset=get_dataset,
        infer=infer,
        post_dist_teardown=post_dist_teardown,
        run_suffix=run_suffix,
    )


def _weight(run: InferRun) -> float:
    return run.model_bundle["model"].weight.item()


def test_latest_checkpoint_is_loaded_in_eval_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events)
    save_dir = _save_dir(tmp_path)
    _write_checkpoint(save_dir / "checkpoint_epoch0_step1_global1", 1.0)
    latest = _write_checkpoint(save_dir / "checkpoint_epoch0_step2_global2", 2.0)

    run = run_experiment(_cfg(tmp_path), _spec(events))

    assert run.checkpoint_dir == str(latest)
    assert _weight(run) == 2.0
    assert not run.model_bundle["model"].training
    assert (run.train_set, run.val_set, run.test_set) == (None, None, None)


def test_explicit_checkpoint_dir_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events)
    _write_checkpoint(_save_dir(tmp_path) / "checkpoint_epoch0_step2_global2", 2.0)
    explicit = _write_checkpoint(tmp_path / "elsewhere", 3.0)

    run = run_experiment(_cfg(tmp_path, checkpoint_dir=str(explicit)), _spec(events))

    assert run.checkpoint_dir == str(explicit)
    assert _weight(run) == 3.0


def test_missing_checkpoint_ends_the_group_without_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events)

    with pytest.raises(FileNotFoundError, match="infer.checkpoint_dir"):
        run_experiment(_cfg(tmp_path), _spec(events))

    assert events == ["end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_checkpoint_keys_must_match_the_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events)
    _write_checkpoint(_save_dir(tmp_path) / "checkpoint_epoch0_step1_global1", 1.0)

    def get_model(cfg: DictConfig) -> torch.nn.Module:
        del cfg
        return torch.nn.Sequential(torch.nn.Linear(1, 1))

    with pytest.raises(RuntimeError):
        run_experiment(_cfg(tmp_path), _spec(events, get_model=get_model))

    assert events == ["end"]


def test_train_hooks_are_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events)
    _write_checkpoint(_save_dir(tmp_path) / "checkpoint_epoch0_step1_global1", 1.0)
    handler = MagicMock()

    run_experiment(_cfg(tmp_path), _spec(events, bundle={"train_hook_handlers": [handler]}))

    handler.disable.assert_called_once_with()


def test_infer_runs_before_teardown_and_feeds_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events)
    _write_checkpoint(_save_dir(tmp_path) / "checkpoint_epoch0_step1_global1", 1.0)

    def infer(run: InferRun) -> float:
        events.append("infer")
        return _weight(run)

    run_experiment(_cfg(tmp_path), _spec(events, infer=infer))

    assert events == ["infer", "wait", "end", "post=1.0"]


def test_datasets_come_from_get_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events)
    _write_checkpoint(_save_dir(tmp_path) / "checkpoint_epoch0_step1_global1", 1.0)

    def get_dataset(cfg: DictConfig) -> tuple[Any, Any, Any]:
        del cfg
        return [0], [1], [2]

    run = run_experiment(_cfg(tmp_path), _spec(events, get_dataset=get_dataset))

    assert (run.train_set, run.val_set, run.test_set) == ([0], [1], [2])


def test_run_suffix_selects_the_checkpoint_and_names_a_new_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    opened: list[dict[str, Any]] = []
    _install(monkeypatch, events)
    _write_checkpoint(_save_dir(tmp_path) / "checkpoint_epoch0_step1_global1", 1.0)
    _write_checkpoint(_save_dir(tmp_path, "-sweep") / "checkpoint_epoch0_step1_global1", 5.0)

    class _Scope:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            opened.append(cast(dict[str, Any], kwargs))

        def __enter__(self) -> MagicMock:
            return MagicMock()

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr("mlkit.util.report.ReporterContext", _Scope)

    run = run_experiment(_cfg(tmp_path), _spec(events, run_suffix="-sweep"))

    assert _weight(run) == 5.0
    assert opened[0]["run_name"] == "toy-toy-lr:0.01-n_steps:1-sweep-infer"
    assert opened[0]["resume_run_id"] is None
