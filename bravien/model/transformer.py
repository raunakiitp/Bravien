"""Normalisation, the residual block, and the transformer stack."""

from __future__ import annotations

import torch
import torch.nn as nn

from bravien.model.attention import BravienAttention, LayerCache
from bravien.model.config import BravienConfig
from bravien.model.embeddings import RotaryEmbedding
from bravien.model.mlp import BravienMLP


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    No mean subtraction and no bias, which makes it cheaper than LayerNorm.
    The reduction runs in float32 even under autocast: computing the inverse
    RMS in bf16 loses enough precision to destabilise training.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(dim=-1, keepdim=True)
        normed = x32 * torch.rsqrt(variance + self.eps)
        return (normed.to(dtype)) * self.weight


def build_norm(config: BravienConfig) -> nn.Module:
    if config.norm_kind == "rmsnorm":
        return RMSNorm(config.hidden_size, eps=config.norm_eps)
    if config.norm_kind == "layernorm":
        return nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
    raise ValueError(f"Unknown norm_kind {config.norm_kind!r}")


class BravienBlock(nn.Module):
    """One pre-norm transformer layer.

    Pre-norm (normalise going *into* each sublayer, add the raw residual back)
    rather than post-norm: it keeps an unnormalised identity path from input to
    output, which is what lets deep stacks train without warmup tricks.
    """

    def __init__(self, config: BravienConfig, rotary: RotaryEmbedding | None) -> None:
        super().__init__()
        self.input_norm = build_norm(config)
        self.attention = BravienAttention(config, rotary)
        self.post_attention_norm = build_norm(config)
        self.mlp = BravienMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        residual = hidden_states
        attn_out, present = self.attention(
            self.input_norm(hidden_states),
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_norm(hidden_states))

        return hidden_states, present
