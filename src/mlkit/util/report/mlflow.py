import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base import Reporter
from .data import _finite_metrics, _plain_data


def _local_sqlite_path(uri: str) -> Path | None:
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return None
    raw = uri[len(prefix) :]
    if raw == "" or raw.startswith(":"):
        return None
    return Path(raw)


def _prepare_sqlite_dir(uri: str) -> None:
    db_path = _local_sqlite_path(uri)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_artifact_dir(artifact_location: str) -> None:
    if "://" not in artifact_location:
        Path(artifact_location).mkdir(parents=True, exist_ok=True)


class MLflowReporter(Reporter):
    """MLflow tracking backend."""

    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment: str,
        run_name: str,
        config: Mapping[str, Any],
        run_id: str | None = None,
        artifact_location: str | None = None,
    ) -> None:
        self.backend = "mlflow"
        self.tracking_uri = tracking_uri or None
        self.experiment = experiment
        self.run_name = run_name
        self.config = config
        self.run_id = run_id
        self.artifact_location = artifact_location or None
        self._auto_close = False
        self._opened = False

    def open(self) -> None:
        import mlflow

        if self.tracking_uri:
            active_uri = self.tracking_uri
        elif env_uri := os.environ.get("MLFLOW_TRACKING_URI"):
            active_uri = env_uri
        else:
            active_uri = "sqlite:///mlflow.db"
        _prepare_sqlite_dir(active_uri)
        if self.tracking_uri or not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(active_uri)
        if self.artifact_location and mlflow.get_experiment_by_name(self.experiment) is None:
            from mlflow.exceptions import MlflowException

            _prepare_artifact_dir(self.artifact_location)
            try:
                mlflow.create_experiment(self.experiment, artifact_location=self.artifact_location)
            except MlflowException as exc:
                if getattr(exc, "error_code", None) != "RESOURCE_ALREADY_EXISTS":
                    raise
        mlflow.set_experiment(self.experiment)

        started_new = False
        try:
            if self.run_id is not None:
                active = mlflow.start_run(run_id=self.run_id, run_name=self.run_name)
                self._auto_close = True
                started_new = True
            elif mlflow.active_run() is None:
                active = mlflow.start_run(run_name=self.run_name)
                self._auto_close = True
                started_new = True
            else:
                active = mlflow.active_run()
                self._auto_close = False

            if started_new and active is not None:
                run_id = active.info.run_id
                if not isinstance(run_id, str) or run_id == "":
                    raise RuntimeError("MLflow did not start a run")
                self.run_id = run_id
                mlflow.log_dict(_plain_data(self.config), "config.yaml")
            elif self.run_id is None and active is not None:
                attached_id = active.info.run_id
                if isinstance(attached_id, str) and attached_id != "":
                    self.run_id = attached_id
        except Exception:
            if started_new and self._auto_close:
                mlflow.end_run(status="FAILED")
            raise
        self._opened = True

    def close(self, *, success: bool) -> None:
        if not self._opened or not self._auto_close:
            return
        import mlflow

        mlflow.end_run(status="FINISHED" if success else "FAILED")
        self._opened = False

    def store_config(self, config: Mapping[str, Any]) -> None:
        self.config = config
        if not self._opened:
            return
        import mlflow

        mlflow.log_dict(_plain_data(self.config), "config.yaml")

    def log(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        metrics = _finite_metrics(data)
        if not metrics:
            return
        import mlflow

        if step is None:
            mlflow.log_metrics(metrics)
        else:
            mlflow.log_metrics(metrics, step=step)