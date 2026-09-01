"""Bravien Unified Model Factory.

Provides dynamic, explicit resolution and instantiation of Bravien models across:
- Native Bravien architectures (Bravien-1.5B, Bravien-v4, Bravien-small, Bravien-tiny)
- Legacy rollback checkpoints (Bravien-v3, Bravien-v2, Bravien-v1)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from bravien.model.bravien_config import BRAVIEN_PRESETS, BravienConfig, get_bravien_preset
from bravien.model.bravien_model import BravienForCausalLM
from bravien.model.checkpoint import load_bravien_checkpoint
from bravien.model.provider import BravienLocalProvider, BravienNativeProvider, ModelProvider
from bravien.utils.logging import get_logger

logger = get_logger("model.factory")


class ModelFactory:
    """Factory resolving Bravien model architectures and providers."""

    @staticmethod
    def create_model(
        model_name_or_preset: str = "bravien-1.5b",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        """Instantiate a model based on preset name or checkpoint directory."""
        dev = torch.device(device)

        # 1. Check if it's a native preset
        if model_name_or_preset in BRAVIEN_PRESETS:
            cfg = get_bravien_preset(model_name_or_preset)
            logger.info("Instantiating native Bravien model preset: %s (%d layers, %d hidden)", cfg.name, cfg.num_layers, cfg.hidden_size)
            model = BravienForCausalLM(cfg)
            if dev.type != "meta":
                model.to(device=dev)
            return model

        # 2. Check if it's a checkpoint path
        ckpt_path = Path(model_name_or_preset)
        if ckpt_path.exists():
            if (ckpt_path / "model.pt").exists() or (ckpt_path / "model_config.json").exists():
                logger.info("Loading native Bravien checkpoint from %s", ckpt_path)
                return load_bravien_checkpoint(ckpt_path, device=dev)
            else:
                # Legacy HF checkpoint (bravien-v1/v2/v3)
                logger.info("Loading legacy rollback checkpoint via HuggingFace: %s", ckpt_path)
                from transformers import AutoModelForCausalLM
                return AutoModelForCausalLM.from_pretrained(
                    str(ckpt_path),
                    torch_dtype=dtype,
                    device_map=dev.type if dev.type == "cuda" else "cpu",
                )

        # Default fallback
        cfg = get_bravien_preset("bravien-1.5b")
        return BravienForCausalLM(cfg).to(device=dev)

    @staticmethod
    def create_provider(
        backend: str | None = None,
        checkpoint_path: str | None = None,
    ) -> ModelProvider:
        """Create the appropriate ModelProvider for runtime inference."""
        chosen_backend = (
            backend or os.environ.get("BRAVIEN_MODEL_BACKEND", "native")
        ).lower()

        if chosen_backend in ("native", "bravien-1.5b", "bravien-v4"):
            ckpt = (
                checkpoint_path
                or os.environ.get("BRAVIEN_CHECKPOINT")
                or os.environ.get("BRAVIEN_MODEL")
                or os.environ.get("BRAVIEN_NATIVE_CHECKPOINT")
                or ("checkpoints/bravien-v4" if Path("checkpoints/bravien-v4").exists() else "checkpoints/bravien-native-1.5b")
            )
            try:
                return BravienNativeProvider(checkpoint_path=ckpt)
            except Exception as e:
                logger.error("Failed to initialize BravienNativeProvider from %s: %s", ckpt, e)
                raise RuntimeError(
                    f"CRITICAL: Failed to initialize native Bravien provider from '{ckpt}'. "
                    f"Error: {e}. Checkpoint validation or tokenizer integrity failed."
                ) from e
        elif chosen_backend in ("qwen_compat", "hf", "legacy", "bravien-v3", "bravien-v2", "bravien-v1"):
            ckpt = checkpoint_path or os.environ.get("BRAVIEN_CHECKPOINT_PATH", "checkpoints/bravien-v3")
            return BravienLocalProvider(checkpoint_path=ckpt)
        else:
            raise ValueError(f"Unsupported model factory backend: '{chosen_backend}'")
