"""Feed-forward network for Bravien (SwiGLU-style gated MLP)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bravien.model.config import BravienConfig

_ACTIVATIONS = {"silu": F.silu, "gelu": F.gelu}


class BravienMLP(nn.Module):
    """Gated feed-forward block.

    Three projections rather than two: `gate` and `up` run in parallel on the
    input, are multiplied elementwise, then projected back down. Biases are
    omitted throughout, which is standard for this family and saves parameters
    without measurable cost.
    """

    def __init__(self, config: BravienConfig) -> None:
        super().__init__()
        if config.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {config.activation!r}; "
                f"expected one of {sorted(_ACTIVATIONS)}"
            )
        self.activation_name = config.activation
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = _ACTIVATIONS[self.activation_name]
        gated = act(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
