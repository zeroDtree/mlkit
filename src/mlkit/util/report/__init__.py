from collections.abc import Mapping
from typing import Any

from .base import NullReporter, Reporter
from .callback import NullReportCallback, ReportCallback
from .mlflow import MLflowReporter
from .run_file import resume_run_id, write_report_run
from .context import ReporterContext
from .wandb import WandbReporter

__all__ = [
    "MLflowReporter",
    "NullReporter",
    "NullReportCallback",
    "ReportCallback",
    "Reporter",
    "ReporterContext",
    "WandbReporter",
    "get_reporter",
    "resume_run_id",
    "write_report_run",
]


def get_reporter(
    kind: str,
    *,
    run_name: str,
    config: Mapping[str, Any],
    mlflow: Mapping[str, Any] | None = None,
    wandb: Mapping[str, Any] | None = None,
    resume_run_id: str | None = None,
) -> Reporter:
    """Build a reporter for the given kind. The run is not open until ``open()``."""
    kind_name = kind.strip().lower()
    if kind_name == "none":
        return NullReporter()
    if kind_name == "mlflow":
        settings = dict(mlflow or {})
        return MLflowReporter(
            tracking_uri=settings.get("tracking_uri") or None,
            experiment=str(settings.get("experiment") or "default"),
            run_name=run_name,
            config=config,
            run_id=resume_run_id,
            artifact_location=settings.get("artifact_location") or None,
        )
    if kind_name == "wandb":
        settings = dict(wandb or {})
        return WandbReporter(
            project=settings.get("project"),
            name=run_name,
            group=settings.get("group"),
            entity=settings.get("entity"),
            mode=settings.get("mode"),
            reinit=bool(settings.get("reinit", True)),
            config=config,
            run_id=resume_run_id,
        )
    raise ValueError(f"Unknown reporter: {kind}")