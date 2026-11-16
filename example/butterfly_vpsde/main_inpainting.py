"""Hydra training example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from utils import get_collate_fn, get_dataset, get_model, inpainting_run_suffix

from mlkit.util.experiment import TrainSpec, run_experiment
from mlkit.util.huggingface import HF_MIRROR  # noqa: F401


@hydra.main(config_path=".", config_name="config_inpainting", version_base=None)
def main(cfg: DictConfig):
    spec = TrainSpec(
        get_model=get_model,
        get_dataset=get_dataset,
        get_collate=get_collate_fn,
        run_suffix=inpainting_run_suffix(cfg),
    )
    run_experiment(cfg, spec)


if __name__ == "__main__":
    main()
