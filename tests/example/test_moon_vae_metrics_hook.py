"""VAE metrics are attached to the model output.

Run with:
    uv run pytest tests/example/test_moon_vae_metrics_hook.py
"""

from __future__ import annotations

from typing import Any

import torch

from example.moon_vae.hooks import build_vae_metrics_hook


def test_vae_metrics_hook_writes_metrics():
    hook = build_vae_metrics_hook(log_interval=2)
    values = torch.tensor([0.0, 1.0])
    model_output: dict[str, Any] = {
        "mu": values,
        "logvar": values,
        "z": values,
        "x_hat": values,
        "recon_loss": torch.tensor(0.5),
        "kl_loss": torch.tensor(0.25),
        "loss": torch.tensor(0.75),
        "kl_weight": 1.0,
    }
    hook(model=None, batch={}, model_output=model_output)
    metrics = model_output["metrics"]
    assert metrics["train/loss"] == 0.75
    assert metrics["train/recon_loss"] == 0.5
    assert metrics["train/kl_loss"] == 0.25
    assert "debug/mu_std" in metrics
    assert "debug/kl_weight" in metrics
