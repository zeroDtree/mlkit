"""Reporter backend protocol and null implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .callback import ReportCallback


class Reporter(ABC):
    """Experiment-tracking backend (MLflow, wandb, or none)."""

    backend: str
    run_id: str | None

    @abstractmethod
    def open(self) -> None:
        """Open or attach to a tracking run."""

    @abstractmethod
    def close(self, *, success: bool) -> None:
        """Close the run when this reporter owns it."""

    @abstractmethod
    def log(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        """Record one metric payload."""

    @abstractmethod
    def store_config(self, config: Mapping[str, Any]) -> None:
        """Replace the stored config for the run."""

    def as_callback(self) -> ReportCallback:
        from .callback import ReportCallback

        return ReportCallback(self)


class NullReporter(Reporter):
    """Reporter that discards metrics and does not open a run."""

    backend = "none"
    run_id: str | None = None

    def open(self) -> None:
        return None

    def close(self, *, success: bool) -> None:
        del success
        return None

    def log(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        del data, step

    def store_config(self, config: Mapping[str, Any]) -> None:
        del config
