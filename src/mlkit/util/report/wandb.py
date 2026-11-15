from collections.abc import Mapping
from typing import Any, Literal

from .base import Reporter
from .data import _plain_data

WandbMode = Literal["online", "offline", "disabled", "shared"]


def _wandb_mode(mode: object) -> WandbMode | None:
    if mode is None:
        return None
    if mode == "online":
        return "online"
    if mode == "offline":
        return "offline"
    if mode == "disabled":
        return "disabled"
    if mode == "shared":
        return "shared"
    raise ValueError(f"Unknown wandb mode: {mode}")


class WandbReporter(Reporter):
    """wandb tracking backend."""

    def __init__(
        self,
        *,
        project: str | None,
        name: str,
        group: str | None,
        entity: str | None,
        mode: str | None,
        reinit: bool,
        config: Mapping[str, Any],
        run_id: str | None = None,
    ) -> None:
        self.backend = "wandb"
        self.project = project
        self.name = name
        self.group = group
        self.entity = entity
        self.mode = mode
        self.reinit = reinit
        self.config = config
        self.run_id = run_id
        self._auto_close = False
        self._opened = False

    def open(self) -> None:
        import wandb

        if self.run_id is not None:
            run = wandb.init(
                reinit=self.reinit,
                mode=_wandb_mode(self.mode),
                project=self.project,
                name=self.name,
                group=self.group,
                entity=self.entity,
                config=_plain_data(self.config),
                id=self.run_id,
                resume="allow",
            )
            self._auto_close = True
        elif wandb.run is None:
            run = wandb.init(
                reinit=self.reinit,
                mode=_wandb_mode(self.mode),
                project=self.project,
                name=self.name,
                group=self.group,
                entity=self.entity,
                config=_plain_data(self.config),
            )
            self._auto_close = True
        else:
            run = wandb.run
            wandb.config.update(_plain_data(self.config), allow_val_change=True)
            self._auto_close = False

        try:
            run_id = None if run is None else run.id
            if not isinstance(run_id, str) or run_id == "":
                raise RuntimeError("wandb did not start a run")
            self.run_id = run_id
        except Exception:
            if self._auto_close:
                wandb.finish(exit_code=1)
            raise
        self._opened = True

    def close(self, *, success: bool) -> None:
        if not self._opened or not self._auto_close:
            return
        import wandb

        wandb.finish(exit_code=0 if success else 1)
        self._opened = False

    def store_config(self, config: Mapping[str, Any]) -> None:
        self.config = config
        if not self._opened:
            return
        import wandb

        wandb.config.update(_plain_data(self.config), allow_val_change=True)

    def log(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        import wandb

        payload = dict(data)
        if step is None:
            wandb.log(payload)
        else:
            wandb.log(payload, step=step)
