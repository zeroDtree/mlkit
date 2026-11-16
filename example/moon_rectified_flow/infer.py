"""Hydra inference example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from utils import get_conditional_model, get_conditional_save_dir, get_model

from mlkit.util.experiment import InferSpec, run_experiment


def _load_conditional_flow(cfg: DictConfig, device):
    from accelerate.utils import SAFE_WEIGHTS_NAME
    from safetensors.torch import load_model

    from mlkit.pipeline.pipeline import latest_checkpoint_dir

    save_dir = get_conditional_save_dir(cfg)
    checkpoint_dir = latest_checkpoint_dir(save_dir)
    if checkpoint_dir is None:
        raise FileNotFoundError(f"No conditional flow checkpoint under {save_dir}; run main.py first")
    pipeline_model = get_conditional_model(cfg)["model"]
    load_model(pipeline_model, os.path.join(checkpoint_dir, SAFE_WEIGHTS_NAME))
    return pipeline_model.to(device).eval().model


def post_dist_teardown(run):
    import matplotlib.pyplot as plt
    import torch
    from torch import Tensor

    cfg = run.cfg
    model_bundle = run.model_bundle
    accelerator = run.accelerator

    model = model_bundle["model"].model
    sampling_hook_handlers = model_bundle["sampling_hook_handlers"]
    n_samples = 256
    n_steps = cfg.flow.n_discretization_steps
    time_steps = torch.linspace(0, 1.0, n_steps + 1)

    # --- Unconditional sampling ---
    for handler in sampling_hook_handlers:
        handler.disable()
    result: dict = model.sampling(shape=(n_samples, 2), device=accelerator.device, return_all=True)
    x_list = [x.detach().cpu() for x in result["x_list"]]

    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    axes[0].scatter(x_list[0][:, 0], x_list[0][:, 1], s=10)
    axes[0].set_title(f"t = {time_steps[0]:.2f}")
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)
    for i in range(n_steps):
        x = x_list[i + 1]
        axes[i + 1].scatter(x[:, 0], x[:, 1], s=10)
        axes[i + 1].set_title(f"t = {time_steps[i + 1]:.2f}")
    fig.tight_layout()
    fig.savefig("sampling_unconditional.png")
    plt.close(fig)

    # --- Classifier-guided conditional sampling ---
    for handler in sampling_hook_handlers:
        handler.enable()
    sigma = 1.0
    x_noise = torch.randn(n_samples, 2) * sigma
    c_eval = torch.randint(0, 2, (n_samples, 1), dtype=torch.float32, device=accelerator.device)
    result_c: dict = model.sampling(
        shape=(n_samples, 2),
        device=accelerator.device,
        return_all=True,
        sampling_condition=c_eval,
    )
    x_list_c = [x.detach().cpu() for x in result_c["x_list"]]
    colors = ["blue" if lbl == 0 else "orange" for lbl in c_eval.squeeze().tolist()]

    fig, axes = plt.subplots(1, n_steps + 1, figsize=(4 * (n_steps + 1), 4), sharex=True, sharey=True)
    axes[0].scatter(x_noise[:, 0], x_noise[:, 1], s=10, c=colors)
    axes[0].set_title(f"t = {time_steps[0]:.2f}")
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)
    for i in range(n_steps):
        x = x_list_c[i + 1]
        axes[i + 1].scatter(x[:, 0], x[:, 1], s=10, c=colors)
        axes[i + 1].set_title(f"t = {time_steps[i + 1]:.2f}")
        axes[i + 1].set_xlim(-3.0, 3.0)
        axes[i + 1].set_ylim(-3.0, 3.0)
    fig.tight_layout()
    fig.savefig("sampling_condition.png")
    plt.close(fig)

    # --- Conditional flow matching ---
    cond_flow = _load_conditional_flow(cfg, accelerator.device)
    c_cfm = torch.randint(0, 2, (n_samples, 1), dtype=torch.float32, device=accelerator.device)
    colors_cfm = ["blue" if lbl == 0 else "orange" for lbl in c_cfm.squeeze().tolist()]

    sampling_result = cond_flow.sampling(
        shape=(n_samples, 2),
        device=accelerator.device,
        return_all=True,
        sampling_condition=c_cfm,
    )
    x_list: list[Tensor] = sampling_result["x_list"]

    cfm_plot_every = 20
    cfm_plot_indices = list(range(0, len(x_list), cfm_plot_every))
    if cfm_plot_indices[-1] != len(x_list) - 1:
        cfm_plot_indices.append(len(x_list) - 1)
    cfm_time_steps = torch.linspace(0, 1.0, len(x_list))

    fig, axes = plt.subplots(
        1,
        len(cfm_plot_indices),
        figsize=(4 * len(cfm_plot_indices), 4),
        sharex=True,
        sharey=True,
    )
    for plot_count, idx in enumerate(cfm_plot_indices):
        x_plot = x_list[idx].detach().cpu()
        axes[plot_count].scatter(x_plot[:, 0], x_plot[:, 1], s=10, c=colors_cfm)
        axes[plot_count].set_title(f"t = {cfm_time_steps[idx]:.2f}")
        axes[plot_count].set_xlim(-3.0, 3.0)
        axes[plot_count].set_ylim(-3.0, 3.0)

    fig.tight_layout()
    fig.savefig("conditional_flow_matching.png")
    plt.close(fig)


SPEC = InferSpec(
    get_model=get_model,
    post_dist_teardown=post_dist_teardown,
)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_experiment(cfg, SPEC)


if __name__ == "__main__":
    main()
