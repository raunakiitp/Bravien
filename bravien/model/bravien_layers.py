"""Native Transformer Decoder Layer for Bravien."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from bravien.model.bravien_attention import BravienAttention, RotaryEmbedding
from bravien.model.bravien_cache import BravienKVCache
from bravien.model.bravien_config import BravienConfig
from bravien.model.bravien_mlp import BravienMLP
from bravien.model.bravien_norm import build_norm


class BravienDecoderLayer(nn.Module):
    """A single Transformer Decoder layer with Pre-RMSNorm and Residual Connections."""

    def __init__(self, config: BravienConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.input_layernorm = build_norm(config)
        self.self_attn = BravienAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = build_norm(config)
        self.mlp = BravienMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: RotaryEmbedding | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: BravienKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # 1. Pre-Norm Self Attention with Residual Connection
        residual = hidden_states
        normed_states = self.input_layernorm(hidden_states)

        attn_output, present_kv = self.self_attn(
            hidden_states=normed_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # 2. Pre-Norm Gated MLP with Residual Connection
        residual = hidden_states
        normed_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_states)
        hidden_states = residual + mlp_output

        return hidden_states, present_kv
