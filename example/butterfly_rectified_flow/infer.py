"""Hydra inference example using :func:`run_experiment`; see ``docs/experiment.md``."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from utils import get_model

from mlkit.util.experiment import InferSpec, run_experiment
from mlkit.util.huggingface import HF_MIRROR  # noqa: F401


def post_dist_teardown(run):
    cfg = run.cfg
    accelerator = run.accelerator
    from diffusers.utils.pil_utils import make_image_grid, numpy_to_pil

    model = run.model_bundle["model"].model

    result = model.sampling(
        shape=(16, 3, cfg.dataset.image_size, cfg.dataset.image_size),
        device=accelerator.device,
    )
    print(f"{result.keys()}")
    image = result["x"]
    print(f"Generated tensor shape: {image.shape}")
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image_grid = make_image_grid(numpy_to_pil(image), rows=4, cols=4)
    save_dir = "fm_image"
    os.makedirs(save_dir, exist_ok=True)
    image_grid.save(os.path.join(save_dir, f"fm_uc_{cfg.flow.n_inference_steps}.png"))


SPEC = InferSpec(
    get_model=get_model,
    post_dist_teardown=post_dist_teardown,
)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_experiment(cfg, SPEC)


if __name__ == "__main__":
    main()
