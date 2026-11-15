"""Reporter open/close lifecycle (attach vs owned run)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from mlkit.util.report import MLflowReporter, ReporterScope, get_reporter


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self._active_run: Any = None

    def set_tracking_uri(self, uri: object) -> None:
        self.calls.append(("uri", uri))

    def set_experiment(self, name: object) -> None:
        self.calls.append(("experiment", name))

    def active_run(self) -> Any:
        return self._active_run

    def start_run(self, run_id: object = None, run_name: object = None) -> Any:
        self.calls.append(("start", run_id, run_name))
        info = type("Info", (), {"run_id": "owned-run"})()
        self._active_run = type("ActiveRun", (), {"info": info})()
        return self._active_run

    def log_dict(self, data: object, artifact_file: object) -> None:
        self.calls.append(("dict", artifact_file, data))

    def end_run(self, status: object = None) -> None:
        self.calls.append(("end", status))
        self._active_run = None


def test_mlflow_attach_does_not_end_external_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    fake = _FakeMlflow()
    external = type("ActiveRun", (), {"info": type("Info", (), {"run_id": "external"})()})()
    fake._active_run = external
    monkeypatch.setitem(sys.modules, "mlflow", cast(Any, fake))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    reporter = get_reporter("mlflow", run_name="run", config={"a": 1}, mlflow={"experiment": "local"})
    reporter.open()
    reporter.close(success=True)

    assert not any(call[0] == "start" for call in fake.calls)
    assert not any(call[0] == "end" for call in fake.calls)
    assert reporter.run_id == "external"


def test_reporter_scope_marks_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", cast(Any, fake))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(RuntimeError, match="boom"):
        with ReporterScope("mlflow", run_name="run", config={}, mlflow={"experiment": "local"}) as reporter:
            del reporter
            raise RuntimeError("boom")

    assert ("end", "FAILED") in fake.calls