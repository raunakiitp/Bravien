"""Native Multi-Head & Grouped-Query Causal Self-Attention with RoPE for Bravien."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from bravien.model.bravien_cache import BravienKVCache, LayerKVCache
from bravien.model.bravien_config import BravienConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Applies complex multiplicative rotations to Q and K representation vectors,
    encoding relative token distances without learned position embedding tables.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Inverse frequencies: shape [dim // 2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute initial cos and sin cache
        self._build_cos_sin_cache(max_position_embeddings)

    def _build_cos_sin_cache(self, max_seq_len: int) -> None:
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._build_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to Query and Key tensors.

    Args:
        q: [batch_size, num_heads, seq_len, head_dim]
        k: [batch_size, num_kv_heads, seq_len, head_dim]
        cos: [seq_len, head_dim] or [batch_size, 1, seq_len, head_dim]
        sin: [seq_len, head_dim] or [batch_size, 1, seq_len, head_dim]
        position_ids: [batch_size, seq_len]
    """
    if position_ids is not None:
        # Index cos and sin with position_ids
        cos = cos[position_ids].unsqueeze(1)  # [bsz, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)  # [bsz, 1, seq_len, head_dim]
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats Key or Value heads when num_kv_heads < num_heads (Grouped Query Attention).

    Input shape: [batch_size, num_kv_heads, seq_len, head_dim]
    Output shape: [batch_size, num_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return x
    bsz, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)
    )


class BravienAttention(nn.Module):
    """Causal Self-Attention with Grouped-Query Attention (GQA) & RoPE."""

    def __init__(self, config: BravienConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_kv_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: RotaryEmbedding | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: BravienKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [bsz, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Calculate rotary embedding offsets if cache is present
        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            kv_seq_len += past_key_values.get_seq_length(self.layer_idx)

        if rotary_emb is not None:
            cos, sin = rotary_emb(value_states, kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        # Update and retrieve from KV Cache if active
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # Repeat KV heads for Grouped Query Attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # FlashAttention / Scaled Dot Product Attention
        dropout_p = self.attention_dropout if self.training else 0.0
        is_causal = attention_mask is None and q_len > 1 and q_len == key_states.shape[-2]

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        else:
            # Fallback manual attention computation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            elif is_causal:
                causal_mask = torch.triu(
                    torch.full((q_len, q_len), float("-inf"), device=hidden_states.device), diagonal=1
                )
                attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if dropout_p > 0.0:
                attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        layer_present = (key_states, value_states) if use_cache else None
        return attn_output, layer_present
