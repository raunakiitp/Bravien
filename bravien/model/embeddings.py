"""Token embeddings and positional encoding for Bravien."""

from __future__ import annotations

import torch
import torch.nn as nn

from bravien.model.config import BravienConfig


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE).

    Produces `cos`/`sin` tables for explicit `position_ids`, so a KV-cached
    decode step can ask for position `n` alone rather than recomputing the
    whole prefix. That explicit-position design is what keeps generation
    consistent with training.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        # Not a parameter: derived constant, but should follow .to(device).
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin), each shaped (B, 1, T, head_dim).

        The head axis is size 1 so the tables broadcast across attention heads.
        """
        # position_ids: (B, T) -> freqs: (B, T, head_dim / 2)
        freqs = position_ids.to(torch.float32)[..., None] * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, T, head_dim)
        cos = emb.cos()[:, None, :, :]
        sin = emb.sin()[:, None, :, :]
        return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: (a, b) -> (-b, a)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to queries and keys.

    Values are deliberately left untouched — RoPE encodes position in the
    query/key dot product only.
    """
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class BravienEmbeddings(nn.Module):
    """Token embedding table, plus learned absolute positions if configured."""

    def __init__(self, config: BravienConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings: nn.Embedding | None = None
        if config.position_encoding == "learned":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.token_embeddings(input_ids)
        if self.position_embeddings is not None:
            if position_ids is None:
                raise ValueError(
                    "position_ids are required when position_encoding='learned'"
                )
            x = x + self.position_embeddings(position_ids)
        return self.dropout(x)
