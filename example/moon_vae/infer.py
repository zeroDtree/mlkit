"""Hydra inference example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from hooks import vae_debug_stats
from omegaconf import DictConfig
from utils import get_dataset, get_model, run_suffix

from mlkit.generative_model.vae import GaussianVAE
from mlkit.util.experiment import InferSpec, run_experiment

COLLAPSE_X_HAT_STD_THRESHOLD = 0.05


def post_dist_teardown(run):
    cfg = run.cfg
    accelerator = run.accelerator
    import matplotlib.pyplot as plt
    import torch

    model: GaussianVAE = run.model_bundle["model"].model

    n_samples = 512
    real_data = run.train_set.data[:n_samples].to(accelerator.device)

    with torch.no_grad():
        samples = model.sampling(shape=(n_samples, 2), device=accelerator.device)
        batch = {"gt_data": real_data, "padding_mask": torch.ones_like(real_data)}
        recon = model.compute_loss(**batch)

    stats = vae_debug_stats(recon, kl_weight=float(recon["kl_weight"]))
    print(f"[VAE debug] {stats}")
    if run.reporter is not None:
        run.reporter.log(stats)
    if stats["debug/x_hat_std"] < COLLAPSE_X_HAT_STD_THRESHOLD:
        print(
            f"[VAE debug] WARNING: possible posterior collapse "
            f"(x_hat_std={stats['debug/x_hat_std']:.4f} < {COLLAPSE_X_HAT_STD_THRESHOLD})"
        )

    real = real_data.cpu().numpy()
    generated = samples["x"].cpu().numpy()
    reconstructed = recon["x_hat"].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    panels = [
        (real, "Real"),
        (generated, "Generated"),
        (reconstructed, "Reconstruction"),
    ]
    for ax, (data, title) in zip(axes, panels, strict=True):
        ax.scatter(data[:, 0], data[:, 1], s=8, alpha=0.6)
        ax.set_title(title)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal")

    fig.tight_layout()
    output_path = f"sampling-kl_weight:{cfg.vae.kl_weight}.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path} (real / generated / reconstruction)")


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
