"""Hydra training example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from shared.model import build_ttur_optimizer
from utils import get_collate_fn, get_dataset, get_model, run_suffix

from mlkit.util.experiment import TrainSpec, run_experiment


def _get_optimizer(model, cfg):
    return build_ttur_optimizer(model.model.generator, model.model.discriminator, cfg)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    spec = TrainSpec(
        get_model=get_model,
        get_dataset=get_dataset,
        get_collate=get_collate_fn,
        get_optimizer=_get_optimizer,
        run_suffix=run_suffix(cfg),
    )
    run_experiment(cfg, spec)


if __name__ == "__main__":
    main()
