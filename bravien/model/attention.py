"""Causal self-attention with grouped-query attention and KV caching."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bravien.model.config import BravienConfig
from bravien.model.embeddings import RotaryEmbedding, apply_rotary_pos_emb

# A cache entry is the (keys, values) pair for one layer, each shaped
# (batch, num_kv_heads, seq_len_so_far, head_dim).
LayerCache = tuple[torch.Tensor, torch.Tensor]


class BravienAttention(nn.Module):
    """Multi-head causal attention with optional grouped queries.

    When `num_kv_heads < num_heads`, several query heads share one key/value
    head. That shrinks the KV cache proportionally, which is the dominant
    memory cost during generation, at a small quality cost.
    """

    def __init__(self, config: BravienConfig, rotary: RotaryEmbedding | None) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = config.num_kv_groups
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.rotary = rotary

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerCache | None]:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE is applied to the *new* tokens using their absolute positions,
        # before they enter the cache. Cached keys were already rotated at the
        # time they were computed, so they must not be rotated again.
        if self.rotary is not None:
            cos, sin = self.rotary(position_ids, dtype=q.dtype)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present: LayerCache | None = (k, v) if use_cache else None

        # Grouped-query attention: replicate each KV head across its group so
        # consecutive query heads share it. repeat_interleave (not repeat) is
        # required — the layout must be [kv0, kv0, kv1, kv1, ...] to line up
        # with query heads [q0, q1, q2, q3, ...].
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        kv_len = k.size(2)
        dropout_p = self.attention_dropout if self.training else 0.0

        if attention_mask is not None:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=dropout_p
            )
        elif q_len == kv_len:
            # No cache: square attention, use the fused causal kernel.
            attn_out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True
            )
        elif q_len == 1:
            # Single-token decode: the one query is the newest position, so it
            # legitimately attends to every cached key. No mask needed.
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            # Multi-token append onto an existing cache needs a bottom-right
            # aligned causal mask. `is_causal=True` would align top-left and
            # silently leak future tokens, so refuse instead of guessing.
            raise ValueError(
                f"An explicit attention_mask is required when q_len ({q_len}) != "
                f"kv_len ({kv_len}) and q_len > 1; is_causal would mis-align."
            )

        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.resid_dropout(self.o_proj(attn_out)), present


def build_causal_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build a boolean attention mask where True means "may attend".

    Aligned bottom-right: query `i` sits at absolute position
    `kv_len - q_len + i`, and may see key `j` when `j <= that position`. This is
    the alignment that makes a cached prefix behave identically to a full
    forward pass.

    `padding_mask` is (B, kv_len) with True on real tokens.
    """
    offset = kv_len - q_len
    rows = torch.arange(q_len, device=device)[:, None] + offset
    cols = torch.arange(kv_len, device=device)[None, :]
    mask = cols <= rows  # (q_len, kv_len)
    mask = mask[None, None, :, :]  # (1, 1, q_len, kv_len)

    if padding_mask is not None:
        # (B, 1, 1, kv_len) — drop padded keys for every query.
        mask = mask & padding_mask[:, None, None, :].to(torch.bool)

    return mask
