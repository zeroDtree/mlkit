"""EMA weights are saved and restored with the module that owns them.

Run with:
    uv run pytest tests/util/test_ema.py
"""

from __future__ import annotations

import torch
from torch import nn

from mlkit.util.ema import EMA


class _Owner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(2, 2)
        self.ema = EMA(self.net, decay=0.5)


def test_shadow_is_part_of_the_owner_state_dict():
    owner = _Owner()

    assert {"ema.shadow.weight", "ema.shadow.bias"} <= set(owner.state_dict())
    assert not any(param.requires_grad for param in owner.ema.parameters())


def test_shadow_survives_a_state_dict_round_trip():
    owner = _Owner()
    with torch.no_grad():
        owner.net.weight.add_(1.0)
    owner.ema.update(owner.net)

    restored = _Owner()
    restored.load_state_dict(owner.state_dict())
    restored.ema.copy_to(restored.net)

    assert torch.equal(restored.net.weight, owner.state_dict()["ema.shadow.weight"])
