"""Hydra inference example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor
from utils import get_dataset, get_model, run_suffix

from mlkit.util.experiment import InferSpec, run_experiment
from mlkit.util.huggingface import HF_MIRROR  # noqa: F401


def post_dist_teardown(run):
    cfg = run.cfg
    accelerator = run.accelerator
    from diffusers.utils.pil_utils import make_image_grid, numpy_to_pil

    model = run.model_bundle["model"].model

    x_0 = torch.stack(run.train_set[0:16]["images"]).to(accelerator.device)
    batch_size, channels, height, width = x_0.shape
    inpainting_mask = torch.zeros(batch_size, channels, height, width, device=accelerator.device)
    inpainting_mask[:, :, :, width // 2 :] = 1.0

    result: Tensor = model.inpainting(
        x=x_0,
        padding_mask=torch.ones(*x_0.shape, device=accelerator.device),
        inpainting_mask=inpainting_mask,
        device=accelerator.device,
        mode=cfg.diffuser.mode,
        n_repaint_steps=cfg.diffuser.n_repaint_steps,
    )
    image = (result / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    image_grid = make_image_grid(numpy_to_pil(image), rows=4, cols=4)
    image_grid.save(
        f"inpainted_sample_{cfg.optimizer.name}_{cfg.diffuser.mode}_{cfg.diffuser.name}_{cfg.diffuser.n_inference_steps}_{cfg.diffuser.n_repaint_steps}.png"
    )


@hydra.main(config_path=".", config_name="config_inpainting", version_base=None)
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
