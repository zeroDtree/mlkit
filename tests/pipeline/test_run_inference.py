"""InferSpec loads trained weights, then runs inference hooks around teardown.

Run with:
    uv run pytest tests/pipeline/test_run_inference.py
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from mlkit.pipeline.callback import CallbackEvent
from mlkit.util.experiment import ConfigSnapshotCallback, InferRun, InferSpec, run_experiment
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


def _write_checkpoint(directory: Path, weight: float, train_cfg: DictConfig | None = None) -> Path:
    from safetensors.torch import save_model

    model = torch.nn.Linear(1, 1)
    torch.nn.init.constant_(model.weight, weight)
    directory.mkdir(parents=True)
    save_model(model, str(directory / "model.safetensors"))
    if train_cfg is not None:
        ConfigSnapshotCallback(train_cfg).on_event(CallbackEvent.PRE_FINALIZE_SAVE, checkpoint_dir=str(directory))
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


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "mlkit.test.infer"]


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


def test_both_hooks_run_under_autocast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events)
    _write_checkpoint(_save_dir(tmp_path) / "checkpoint_epoch0_step1_global1", 1.0)

    @contextmanager
    def autocast() -> Iterator[None]:
        events.append("autocast>")
        yield
        events.append("<autocast")

    accelerator.autocast.side_effect = autocast

    def infer(run: InferRun) -> int:
        del run
        events.append("infer")
        return 1

    run_experiment(_cfg(tmp_path), _spec(events, infer=infer))

    assert events == ["autocast>", "infer", "<autocast", "wait", "end", "autocast>", "post=1", "<autocast"]


def test_config_snapshot_is_resolved_when_the_checkpoint_is_saved(tmp_path: Path):
    cfg = OmegaConf.create({"model": {"width": 4, "copy": "${model.width}"}})
    callback = ConfigSnapshotCallback(cfg)

    callback.on_event(CallbackEvent.POST_SAVE, checkpoint_dir=str(tmp_path))
    assert not (tmp_path / "config.yaml").exists()

    cfg.model.width = 8
    callback.on_event(CallbackEvent.PRE_FINALIZE_SAVE, checkpoint_dir=str(tmp_path))
    assert OmegaConf.load(tmp_path / "config.yaml") == {"model": {"width": 8, "copy": 8}}


def test_model_side_config_drift_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    events: list[str] = []
    _install(monkeypatch, events)
    train_cfg = _cfg(tmp_path)
    train_cfg.model.width = 8
    train_cfg.train.n_steps = 999
    train_cfg.optimizer.lr = 0.5
    train_cfg.sampling = {"steps": 3}
    checkpoint = _write_checkpoint(tmp_path / "ckpt", 1.0, train_cfg)
    cfg = _cfg(tmp_path, checkpoint_dir=str(checkpoint))
    cfg.model.width = 16

    with caplog.at_level(logging.WARNING, logger="mlkit.test.infer"):
        run = run_experiment(cfg, _spec(events))

    assert run.train_cfg is not None
    assert run.train_cfg.model.width == 8
    assert _messages(caplog) == [
        "Inference config differs from the training config:\n"
        "model.width: train=8, infer=16\n"
        "sampling.steps: train=3, infer='<unset>'"
    ]


def test_matching_config_logs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    events: list[str] = []
    _install(monkeypatch, events)
    _write_checkpoint(_save_dir(tmp_path) / "checkpoint_epoch0_step1_global1", 1.0, _cfg(tmp_path))

    with caplog.at_level(logging.WARNING, logger="mlkit.test.infer"):
        run_experiment(_cfg(tmp_path), _spec(events))

    assert _messages(caplog) == []


def test_checkpoint_without_snapshot_skips_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    events: list[str] = []
    _install(monkeypatch, events)
    _write_checkpoint(_save_dir(tmp_path) / "checkpoint_epoch0_step1_global1", 1.0)

    with caplog.at_level(logging.WARNING, logger="mlkit.test.infer"):
        run = run_experiment(_cfg(tmp_path), _spec(events))

    assert run.train_cfg is None
    assert _messages(caplog) == [f"{run.checkpoint_dir} has no config.yaml; the config check is skipped"]


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
