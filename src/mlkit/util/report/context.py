"""Reporter lifecycle context manager (not :class:`mlkit.util.experiment.TrainRunContext`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Reporter


class ReporterContext:
    """Class context manager: ``open`` on enter, ``close`` on exit."""

    def __init__(
        self,
        kind: str,
        *,
        run_name: str,
        config: Mapping[str, Any],
        mlflow: Mapping[str, Any] | None = None,
        wandb: Mapping[str, Any] | None = None,
        resume_run_id: str | None = None,
    ) -> None:
        from . import get_reporter

        self._reporter = get_reporter(
            kind,
            run_name=run_name,
            config=config,
            mlflow=mlflow,
            wandb=wandb,
            resume_run_id=resume_run_id,
        )
        self._opened = False

    def __enter__(self) -> Reporter:
        self._reporter.open()
        self._opened = True
        return self._reporter

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del tb
        if self._opened:
            self._reporter.close(success=exc_type is None)
        return None
