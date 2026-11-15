"""Hydra training example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from hooks import vae_debug_stats
from omegaconf import DictConfig
from utils import get_collate_fn, get_dataset, get_model

from mlkit.generative_model.vae import GaussianVAE
from mlkit.util.experiment import ExperimentSpec, run_experiment

COLLAPSE_X_HAT_STD_THRESHOLD = 0.05


def post_dist_teardown(ctx):
    cfg = ctx.cfg
    pipeline = ctx.pipeline
    accelerator = ctx.accelerator
    train_set = ctx.train_set
    del ctx.train_set
    import matplotlib.pyplot as plt
    import torch

    model: GaussianVAE = get_model(
        cfg,
        final_model_ckpt_path=f"{pipeline.get_latest_checkpoint_dir()}/model.safetensors",
    )["model"].model
    model = model.to(accelerator.device)
    model.eval()

    n_samples = 512
    real_data = train_set.data[:n_samples].to(accelerator.device)

    with torch.no_grad():
        samples = model.sampling(shape=(n_samples, 2), device=accelerator.device)
        batch = {"gt_data": real_data, "padding_mask": torch.ones_like(real_data)}
        recon = model.compute_loss(**batch)

    stats = vae_debug_stats(recon, kl_weight=float(recon["kl_weight"]))
    print(f"[VAE debug] {stats}")
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
    spec = ExperimentSpec(
        get_model=get_model,
        get_dataset=get_dataset,
        get_collate=get_collate_fn,
        post_dist_teardown=post_dist_teardown,
        save_dir_suffix=f"-kl_weight:{cfg.vae.kl_weight}",
    )
    run_experiment(cfg, spec)


if __name__ == "__main__":
    main()
