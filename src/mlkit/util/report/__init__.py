from collections.abc import Mapping
from typing import Any

from .callback import NullReportCallback, ReportCallback
from .mlflow import MLflowCallback
from .run_file import resume_run_id, write_report_run
from .wandb import WandbCallback

__all__ = [
    "MLflowCallback",
    "NullReportCallback",
    "ReportCallback",
    "WandbCallback",
    "reporter_callback",
    "resume_run_id",
    "write_report_run",
]


def reporter_callback(
    kind: str,
    *,
    run_name: str,
    config: Mapping[str, Any],
    mlflow: Mapping[str, Any] | None = None,
    wandb: Mapping[str, Any] | None = None,
    resume_run_id: str | None = None,
) -> ReportCallback:
    """Build the report callback selected by ``kind``.

    The returned callback opens its run when entered and closes it when the
    context exits. ``"none"`` returns a callback that discards every payload.

    Args:
        kind: ``"mlflow"``, ``"wandb"``, or ``"none"``.
        run_name: run name passed to the selected backend.
        config: resolved config stored with the run.
        mlflow: MLflow settings with ``tracking_uri``, ``experiment``, and
            optional ``artifact_location``.
        wandb: wandb settings with ``project``, ``group``, ``entity``, ``mode``,
            and ``reinit``.
        resume_run_id: existing run to continue. ``None`` starts a new run.

    Returns:
        A callback that has not opened its run yet.

    Raises:
        ValueError: ``kind`` is not one of the known reporters.
    """
    kind_name = kind.strip().lower()
    if kind_name == "none":
        return NullReportCallback()
    if kind_name == "mlflow":
        settings = dict(mlflow or {})
        return MLflowCallback(
            tracking_uri=settings.get("tracking_uri") or None,
            experiment=str(settings.get("experiment") or "default"),
            run_name=run_name,
            config=config,
            run_id=resume_run_id,
            artifact_location=settings.get("artifact_location") or None,
        )
    if kind_name == "wandb":
        settings = dict(wandb or {})
        return WandbCallback(
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
