"""Dynamic KV Cache for efficient autoregressive generation in Bravien."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


LayerKVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class BravienKVCache:
    """Manages Key-Value states across all Transformer layers during generation."""

    key_cache: list[torch.Tensor] = field(default_factory=list)
    value_cache: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def create_empty(cls, num_layers: int) -> BravienKVCache:
        return cls(
            key_cache=[torch.empty(0)] * num_layers,
            value_cache=[torch.empty(0)] * num_layers,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new key and value states to the cache for the specified layer.

        Args:
            key_states: [batch_size, num_kv_heads, seq_len, head_dim]
            value_states: [batch_size, num_kv_heads, seq_len, head_dim]
            layer_idx: Index of transformer layer

        Returns:
            Tuple of full (cached + new) (keys, values)
        """
        if self.key_cache[layer_idx].numel() == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current sequence length in the cache."""
        if len(self.key_cache) > layer_idx and self.key_cache[layer_idx].numel() > 0:
            return self.key_cache[layer_idx].shape[2]
        return 0

    def reset(self) -> None:
        for i in range(len(self.key_cache)):
            self.key_cache[i] = torch.empty(0)
            self.value_cache[i] = torch.empty(0)

    def memory_footprint_bytes(self) -> int:
        total = 0
        for k, v in zip(self.key_cache, self.value_cache):
            if k.numel() > 0:
                total += k.element_size() * k.numel()
            if v.numel() > 0:
                total += v.element_size() * v.numel()
        return total
