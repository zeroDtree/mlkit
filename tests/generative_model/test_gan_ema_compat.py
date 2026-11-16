"""GAN checkpoints without EMA shadow weights still load under strict safetensors.

Run with:
    uv run pytest tests/generative_model/test_gan_ema_compat.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from safetensors.torch import load_model, save_file
from torch import Tensor
from torch.nn import Module, Sequential

from mlkit.generative_model.gan import GAN, GANConfig


class _Generator(Module):
    def __init__(self, latent_dim: int, data_dim: int) -> None:
        super().__init__()
        self.net = Sequential(
            torch.nn.Linear(latent_dim, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, data_dim),
        )

    def forward(self, **batch: object) -> dict[str, Tensor]:
        return {"x": self.net(batch["z"])}


class _Discriminator(Module):
    def __init__(self, data_dim: int) -> None:
        super().__init__()
        self.net = Sequential(
            torch.nn.Linear(data_dim, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 1),
        )

    def forward(self, **batch: object) -> dict[str, Tensor]:
        return {"logits": self.net(batch["gt_data"]).squeeze(-1)}


class _Wrapper(Module):
    def __init__(self, gan: GAN) -> None:
        super().__init__()
        self.model = gan


def _gan() -> GAN:
    latent_dim, data_dim = 4, 16
    config = GANConfig(ndim_micro_shape=1, latent_dim=latent_dim, n_critic=1, ema_decay=0.999)
    return GAN(
        config=config,
        generator=_Generator(latent_dim, data_dim),
        discriminator=_Discriminator(data_dim),
    )


def test_legacy_checkpoint_fills_ema_and_warns(tmp_path: Path):
    wrapper = _Wrapper(_gan())
    legacy = {key: tensor for key, tensor in wrapper.state_dict().items() if not key.startswith("model.ema.shadow.")}
    path = tmp_path / "legacy.safetensors"
    save_file(legacy, str(path))

    loaded = _Wrapper(_gan())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_model(loaded, str(path), strict=True)

    assert len(caught) == 1
    assert "EMA shadow" in str(caught[0].message)
    assert loaded.model.ema is not None
    for s_param, g_param in zip(
        loaded.model.ema.shadow.parameters(),
        loaded.model.generator.parameters(),
        strict=True,
    ):
        assert torch.equal(s_param, g_param)


def test_full_checkpoint_loads_without_warning(tmp_path: Path):
    wrapper = _Wrapper(_gan())
    path = tmp_path / "full.safetensors"
    save_file(wrapper.state_dict(), str(path))

    loaded = _Wrapper(_gan())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_model(loaded, str(path), strict=True)

    assert not any("EMA shadow" in str(w.message) for w in caught)
