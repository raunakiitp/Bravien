"""Parameter Counting Engine for Bravien Transformer Architectures."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from bravien.model.bravien_config import BravienConfig


def count_parameters(model: nn.Module) -> dict[str, Any]:
    """Calculate exact parameter counts from an instantiated PyTorch model or config."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Sub-component categorization
    embedding_params = 0
    attention_params = 0
    mlp_params = 0
    norm_params = 0
    lm_head_params = 0

    for name, param in model.named_parameters():
        num = param.numel()
        if "embed_tokens" in name:
            embedding_params += num
        elif "self_attn" in name:
            attention_params += num
        elif "mlp" in name:
            mlp_params += num
        elif "norm" in name:
            norm_params += num
        elif "lm_head" in name:
            lm_head_params += num

    # Memory calculations
    mem_fp32_gb = (total_params * 4) / (1024**3)
    mem_bf16_gb = (total_params * 2) / (1024**3)
    mem_int8_gb = (total_params * 1) / (1024**3)
    mem_int4_gb = (total_params * 0.5) / (1024**3)

    is_1p5b_range = 1_350_000_000 <= total_params <= 1_650_000_000
    is_494m_range = 450_000_000 <= total_params <= 550_000_000

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "total_billions": round(total_params / 1e9, 3),
        "total_millions": round(total_params / 1e6, 2),
        "embedding_parameters": embedding_params,
        "attention_parameters": attention_params,
        "mlp_parameters": mlp_params,
        "norm_parameters": norm_params,
        "lm_head_parameters": lm_head_params,
        "memory_weights_gb": {
            "fp32": round(mem_fp32_gb, 2),
            "bf16": round(mem_bf16_gb, 2),
            "int8": round(mem_int8_gb, 2),
            "int4": round(mem_int4_gb, 2),
        },
        "is_target_1p5b": is_1p5b_range,
        "is_legacy_494m": is_494m_range,
        "status": "PASS" if is_1p5b_range else ("LEGACY_BASELINE" if is_494m_range else "OTHER"),
    }
