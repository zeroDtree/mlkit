from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, cast

from mlkit.pipeline.callback import BaseCallback, CallbackEvent

from .run_file import write_report_run


class ReportCallback(BaseCallback):
    """Experiment-tracking callback.

    ``log`` writes one metric payload. ``store_config`` replaces the config
    stored with the run. Entering the callback opens the run, and leaving it
    closes the run.
    """

    backend: str = "none"
    run_id: str | None = None

    def on_event(self, event: CallbackEvent, *_args: object, **kwargs: object) -> None:
        """Record a log payload or the run id of a checkpoint.

        Args:
            event: ``LOG`` writes metrics. ``PRE_FINALIZE_SAVE`` writes the run id
                into ``checkpoint_dir`` when a run is open.
            *_args: unused positional arguments from the pipeline.
            **kwargs: ``metrics`` and ``step`` for ``LOG``. ``checkpoint_dir``
                for ``PRE_FINALIZE_SAVE``.
        """
        if event is CallbackEvent.LOG:
            metrics = kwargs.get("metrics")
            if not isinstance(metrics, Mapping):
                return
            step = kwargs.get("step")
            self.log(cast(Mapping[str, Any], metrics), step=step if isinstance(step, int) else None)
            return
        if event is not CallbackEvent.PRE_FINALIZE_SAVE:
            return
        directory = kwargs.get("checkpoint_dir")
        if not isinstance(directory, str) or not isinstance(self.backend, str):
            return
        if not isinstance(self.run_id, str) or self.run_id == "":
            return
        write_report_run(directory, backend=self.backend, run_id=self.run_id)

    @abstractmethod
    def log(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        """Record one payload.

        Args:
            data: values to record.
            step: optional step index for this payload.
        """
        ...

    @abstractmethod
    def store_config(self, config: Mapping[str, Any]) -> None:
        """Replace the config stored with the run.

        Args:
            config: resolved config after the scheduler has rewritten it.
        """
        ...

    @abstractmethod
    def __enter__(self) -> "ReportCallback":
        """Open the run and return this callback."""
        ...

    @abstractmethod
    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the run. A raised exception marks the run failed."""
        ...


class NullReportCallback(ReportCallback):
    """Callback that discards metrics and does not open a run."""

    backend = "none"

    def log(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        """Discard the payload."""

    def store_config(self, config: Mapping[str, Any]) -> None:
        """Discard a replacement config."""

    def __enter__(self) -> "NullReportCallback":
        """Return this callback without opening a run."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Leave the callback without closing a run."""
