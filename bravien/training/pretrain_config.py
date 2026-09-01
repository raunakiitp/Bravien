"""Pretraining configuration for native Bravien Transformer models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class PretrainConfig:
    """Hyperparameters and execution configuration for pretraining Bravien models."""

    # Optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    max_tokens: int | None = None
    initial_step: int = 0
    initial_tokens: int = 0
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Batching & Context
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 2048

    # Precision & Hardware
    mixed_precision: Literal["bf16", "fp16", "no"] = "bf16"
    gradient_checkpointing: bool = False
    device: str = "auto"
    seed: int = 42

    # Logging & Checkpointing
    log_interval: int = 10
    eval_interval: int = 100
    eval_steps: int = 20
    save_interval: int = 250
    keep_last_n_checkpoints: int = 3
    output_dir: str = "checkpoints/native_pretrain"
    resume_from_checkpoint: str | None = None

    # Smoke / Debug Mode
    tiny_smoke_mode: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PretrainConfig:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PretrainConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
