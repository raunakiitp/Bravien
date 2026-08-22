"""Optimizer construction (§22).

The only real decision here is which parameters get weight decay. Decaying a
LayerNorm/RMSNorm gain or a bias pulls it toward zero, which fights the
normalisation it exists to provide, so those are separated into their own group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    #: Decoupled AdamW is the default; SGD exists for tiny CPU experiments.
    kind: str = "adamw"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        for name, beta in (("beta1", self.beta1), ("beta2", self.beta2)):
            if not 0 <= beta < 1:
                raise ValueError(f"{name} must be in [0, 1), got {beta}")
        if self.kind not in ("adamw", "sgd"):
            raise ValueError(f"unknown optimizer kind {self.kind!r}")


def split_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition into (decay, no_decay).

    Rule: tensors with 2 or more dimensions are weight matrices and get decay;
    1-D tensors are norms and biases and do not. Tied parameters appear once
    because they are de-duplicated by identity.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (decay if param.dim() >= 2 else no_decay).append(param)

    return decay, no_decay


def build_optimizer(
    model: nn.Module, config: OptimizerConfig, *, fused: bool | None = None
) -> torch.optim.Optimizer:
    """Build the optimizer with correct weight-decay grouping.

    Args:
        fused: use the fused CUDA implementation. None auto-selects it when a
            CUDA device is present, where it is a straightforward speedup.
    """
    decay, no_decay = split_parameters(model)
    if not decay and not no_decay:
        raise ValueError("model has no trainable parameters")

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if config.kind == "sgd":
        return torch.optim.SGD(groups, lr=config.learning_rate, momentum=0.9)

    if fused is None:
        fused = torch.cuda.is_available()

    kwargs: dict[str, object] = {
        "lr": config.learning_rate,
        "betas": (config.beta1, config.beta2),
        "eps": config.eps,
    }
    if fused:
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(groups, **kwargs)  # type: ignore[arg-type]
    except (RuntimeError, ValueError):
        # Fused AdamW rejects some dtype/device combinations; the unfused path is
        # numerically identical, just slower.
        kwargs.pop("fused", None)
        return torch.optim.AdamW(groups, **kwargs)  # type: ignore[arg-type]


def parameter_group_summary(optimizer: torch.optim.Optimizer) -> str:
    lines = [f"{type(optimizer).__name__}"]
    for index, group in enumerate(optimizer.param_groups):
        count = sum(p.numel() for p in group["params"])
        lines.append(
            f"  group {index}: {len(group['params'])} tensors, {count:,} params, "
            f"lr={group['lr']:.2e}, weight_decay={group.get('weight_decay', 0.0)}"
        )
    return "\n".join(lines)
