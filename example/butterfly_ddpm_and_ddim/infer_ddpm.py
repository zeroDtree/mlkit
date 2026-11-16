"""Hydra inference example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from utils import get_model, run_suffix

from mlkit.util.experiment import InferSpec, run_experiment
from mlkit.util.huggingface import HF_MIRROR  # noqa: F401


def on_finish(run):
    cfg = run.cfg
    accelerator = run.accelerator
    from diffusers.utils.pil_utils import make_image_grid, numpy_to_pil

    model = run.model_bundle["model"].model

    result: dict = model.sampling(
        shape=(16, 3, cfg.dataset.image_size, cfg.dataset.image_size),
        device=accelerator.device,
        mode=cfg.diffuser.mode,
    )
    image = result["x"]
    print(f"Generated tensor shape: {image.shape}")
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.float().cpu().permute(0, 2, 3, 1).numpy()
    image_grid = make_image_grid(numpy_to_pil(image), rows=4, cols=4)
    image_grid.save(f"generated_sample_{cfg.optimizer.name}_{cfg.diffuser.mode}_{cfg.diffuser.name}.png")

    E_x0_xt_list = result["E_x0_xt_list"]
    if E_x0_xt_list is not None and len(E_x0_xt_list) > 0:
        num_samples = min(8, len(E_x0_xt_list))
        indices = (
            [int(i * (len(E_x0_xt_list) - 1) / (num_samples - 1)) for i in range(num_samples)]
            if num_samples > 1
            else [0]
        )
        sampled_images = []
        for idx in indices:
            img = (E_x0_xt_list[idx][0:1] / 2 + 0.5).clamp(0, 1)
            sampled_images.extend(numpy_to_pil(img.float().cpu().permute(0, 2, 3, 1).numpy()))
        grid_rows = 2 if num_samples > 4 else 1
        grid_cols = (num_samples + grid_rows - 1) // grid_rows
        make_image_grid(sampled_images, rows=grid_rows, cols=grid_cols).save(
            f"denoising_process_{cfg.optimizer.name}_{cfg.diffuser.mode}_{cfg.diffuser.name}.png"
        )
        print(f"Saved denoising process ({num_samples} of {len(E_x0_xt_list)} steps)")


@hydra.main(config_path=".", config_name="config_ddpm", version_base=None)
def main(cfg: DictConfig):
    spec = InferSpec(
        get_model=get_model,
        on_finish=on_finish,
        run_suffix=run_suffix(cfg),
    )
    run_experiment(cfg, spec)


if __name__ == "__main__":
    main()
