"""Native Gated MLP / SwiGLU Feed-Forward Network for Bravien."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bravien.model.bravien_config import BravienConfig


class BravienMLP(nn.Module):
    """SwiGLU Gated Feed-Forward Network (Shazeer, 2020).

    Computes: down_proj(act(gate_proj(x)) * up_proj(x))
    This provides significantly better capacity-per-parameter compared to standard 2-layer MLPs.
    """

    def __init__(self, config: BravienConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        if config.activation == "silu":
            self.act_fn = F.silu
        elif config.activation == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: '{config.activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated Linear Unit with SiLU activation
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Semantic alias
BravienSwiGLUMLP = BravienMLP

