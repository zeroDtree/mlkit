"""Hydra inference example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from utils import get_dataset, get_model, run_suffix

from mlkit.generative_model.gan import GAN
from mlkit.util.experiment import InferSpec, run_experiment


def post_dist_teardown(run):
    cfg = run.cfg
    accelerator = run.accelerator
    import matplotlib.pyplot as plt

    gan: GAN = run.model_bundle["model"].model
    if gan.ema is not None:
        gan.ema.copy_to(gan.generator)

    n_samples = 512
    real_data = run.train_set.data[:n_samples].cpu().numpy()

    result = gan.sampling(
        shape=(n_samples, 2),
        device=accelerator.device,
    )
    generated = result["x"].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    for ax, data, title in zip(
        axes,
        [real_data, generated],
        ["Real", "Generated (EMA)"],
        strict=True,
    ):
        ax.scatter(data[:, 0], data[:, 1], s=8, alpha=0.6)
        ax.set_title(title)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"sampling-latent:{cfg.gan.latent_dim}.png")
    plt.close(fig)
    print(f"Saved sampling-latent:{cfg.gan.latent_dim}.png")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    spec = InferSpec(
        get_model=get_model,
        get_dataset=get_dataset,
        post_dist_teardown=post_dist_teardown,
        run_suffix=run_suffix(cfg),
    )
    run_experiment(cfg, spec)


if __name__ == "__main__":
    main()
