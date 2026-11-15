import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from shared.model import build_ttur_optimizer
from shared.runner import ExperimentSpec, run_experiment
from utils import get_collate_fn, get_dataset, get_model

from mlkit.generative_model.gan import GAN


def post_dist_teardown(ctx):
    cfg = ctx.cfg
    model_bundle = ctx.model_bundle
    pipeline = ctx.pipeline
    accelerator = ctx.accelerator
    del ctx.train_set
    import matplotlib.pyplot as plt

    gan: GAN = model_bundle["model"].model
    gan = gan.to(accelerator.device)
    gan.eval()
    if gan.ema is not None:
        gan.ema.copy_to(gan.generator)

    n_samples = 512
    real_data = train_set.data[:n_samples].cpu().numpy()

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


def _get_optimizer(model, cfg):
    return build_ttur_optimizer(model.model.generator, model.model.discriminator, cfg)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    spec = ExperimentSpec(
        get_model=get_model,
        get_dataset=get_dataset,
        get_collate=get_collate_fn,
        get_optimizer=_get_optimizer,
        post_dist_teardown=post_dist_teardown,
        save_dir_suffix=f"-latent:{cfg.gan.latent_dim}",
    )
    run_experiment(cfg, spec)


if __name__ == "__main__":
    main()
