"""Hydra training example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from utils import get_collate_fn, get_dataset, get_model

from mlkit.util.experiment import TrainSpec, run_experiment

SPEC = TrainSpec(
    get_model=get_model,
    get_dataset=get_dataset,
    get_collate=get_collate_fn,
    disable_train_hooks=True,
)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_experiment(cfg, SPEC)


if __name__ == "__main__":
    main()
