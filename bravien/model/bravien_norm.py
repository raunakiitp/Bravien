"""Native normalization layers for Bravien Transformer architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from bravien.model.bravien_config import BravienConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Computes RMS without subtracting mean, improving throughput while maintaining
    numerical stability in bfloat16 and float16 training.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability, then cast back
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"


def build_norm(config: BravienConfig) -> nn.Module:
    """Instantiate the configured normalization layer."""
    if config.norm_kind == "rmsnorm":
        return RMSNorm(config.hidden_size, eps=config.norm_eps)
    elif config.norm_kind == "layernorm":
        return nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
    else:
        raise ValueError(f"Unknown norm kind: '{config.norm_kind}'")


# Semantic alias
BravienRMSNorm = RMSNorm
