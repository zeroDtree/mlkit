import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from mlkit.util.experiment import ExperimentSpec, run_experiment
from utils import get_collate_fn, get_dataset, get_model

from mlkit.util.huggingface import HF_MIRROR  # noqa: F401


def post_dist_teardown(ctx):
    cfg = ctx.cfg
    model_bundle = ctx.model_bundle
    pipeline = ctx.pipeline
    accelerator = ctx.accelerator
    del ctx.train_set
    from diffusers.utils.pil_utils import make_image_grid, numpy_to_pil

    model = get_model(
        cfg,
        final_model_ckpt_path=f"{pipeline.get_latest_checkpoint_dir()}/model.safetensors",
    )["model"].model
    model = model.to(accelerator.device)

    result: dict = model.sampling(
        shape=(16, 3, cfg.dataset.image_size, cfg.dataset.image_size),
        device=accelerator.device,
    )
    image = result["x"]
    print(f"Generated tensor shape: {image.shape}")
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image_grid = make_image_grid(numpy_to_pil(image), rows=4, cols=4)
    image_grid.save(
        f"generated_sample_{cfg.diffusion.name}_{cfg.diffusion.n_inference_steps}_{cfg.diffusion.n_correct_steps}.png"
    )


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    spec = ExperimentSpec(
        get_model=get_model,
        get_dataset=get_dataset,
        get_collate=get_collate_fn,
        post_dist_teardown=post_dist_teardown,
        save_dir_suffix=f"-{cfg.optimizer.name}",
    )
    run_experiment(cfg, spec)


if __name__ == "__main__":
    main()
