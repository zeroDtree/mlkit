from collections.abc import Mapping
from typing import Any, cast

from mlkit.pipeline.callback import BaseCallback, CallbackEvent

from .base import Reporter, NullReporter
from .run_file import write_report_run


class ReportCallback(BaseCallback):
    """Forwards pipeline events to a :class:`Reporter`."""

    def __init__(self, reporter: Reporter) -> None:
        self._reporter = reporter

    @property
    def backend(self) -> str:
        return self._reporter.backend

    @backend.setter
    def backend(self, value: str) -> None:
        self._reporter.backend = value

    @property
    def run_id(self) -> str | None:
        return self._reporter.run_id

    @run_id.setter
    def run_id(self, value: str | None) -> None:
        self._reporter.run_id = value

    def on_event(self, event: CallbackEvent, *_args: object, **kwargs: object) -> None:
        if event is CallbackEvent.LOG:
            metrics = kwargs.get("metrics")
            if not isinstance(metrics, Mapping):
                return
            step = kwargs.get("step")
            self._reporter.log(cast(Mapping[str, Any], metrics), step=step if isinstance(step, int) else None)
            return
        if event is not CallbackEvent.PRE_FINALIZE_SAVE:
            return
        directory = kwargs.get("checkpoint_dir")
        if not isinstance(directory, str):
            return
        run_id = self._reporter.run_id
        if not isinstance(run_id, str) or run_id == "":
            return
        write_report_run(directory, backend=self._reporter.backend, run_id=run_id)

    def log(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        self._reporter.log(data, step=step)

    def store_config(self, config: Mapping[str, Any]) -> None:
        self._reporter.store_config(config)


class NullReportCallback(ReportCallback):
    """Pipeline callback backed by :class:`NullReporter`."""

    def __init__(self) -> None:
        super().__init__(NullReporter())