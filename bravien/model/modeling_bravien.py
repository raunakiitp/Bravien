"""Bravien Native Causal Language Model Architecture.

Pure PyTorch implementation of the sovereign ~1.5B parameter Bravien Transformer:
- 32 Decoder Blocks
- 4:1 Grouped-Query Attention (GQA)
- SwiGLU Gated Feed-Forward Networks
- Rotary Position Embeddings (RoPE)
- Pre-RMSNorm
- Dynamic Key-Value Caching
"""

from __future__ import annotations

from bravien.model.bravien_attention import BravienAttention, RotaryEmbedding
from bravien.model.bravien_cache import BravienKVCache
from bravien.model.bravien_config import BravienConfig
from bravien.model.bravien_layers import BravienDecoderLayer
from bravien.model.bravien_mlp import BravienMLP, BravienSwiGLUMLP
from bravien.model.bravien_model import BravienForCausalLM, BravienModel, CausalLMOutput
from bravien.model.bravien_norm import BravienRMSNorm, RMSNorm

__all__ = [
    "BravienConfig",
    "BravienModel",
    "BravienForCausalLM",
    "BravienDecoderLayer",
    "BravienAttention",
    "RotaryEmbedding",
    "BravienMLP",
    "BravienSwiGLUMLP",
    "BravienRMSNorm",
    "RMSNorm",
    "BravienKVCache",
    "CausalLMOutput",
]
