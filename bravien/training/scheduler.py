"""Learning-rate schedules (§22).

Implemented as a plain function of step count rather than a `torch.optim.
lr_scheduler` subclass. The state that needs checkpointing is then a single
integer, so resuming cannot desynchronise the schedule from the optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    """Warmup followed by decay.

    Warmup matters more than the decay shape at small scale: Adam's second-moment
    estimate is unreliable for the first few hundred steps, and a full learning
    rate applied then is the most common cause of an early loss spike.
    """

    warmup_steps: int = 100
    total_steps: int = 1000
    min_lr_ratio: float = 0.1
    kind: str = "cosine"

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps >= self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be less than "
                f"total_steps ({self.total_steps})"
            )
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.kind not in ("cosine", "linear", "constant"):
            raise ValueError(f"unknown schedule kind {self.kind!r}")


def lr_at_step(step: int, base_lr: float, config: SchedulerConfig) -> float:
    """Learning rate for a given step, 0-indexed.

    Step `config.warmup_steps` is the first at full `base_lr`; steps at or past
    `total_steps` hold the floor rather than going negative.
    """
    if step < 0:
        raise ValueError("step must be non-negative")

    if config.warmup_steps > 0 and step < config.warmup_steps:
        # Start at 1/warmup rather than 0, so step 0 makes some progress.
        return base_lr * (step + 1) / config.warmup_steps

    if config.kind == "constant":
        return base_lr

    decay_steps = config.total_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / max(decay_steps, 1)
    progress = min(max(progress, 0.0), 1.0)

    floor = base_lr * config.min_lr_ratio
    if config.kind == "linear":
        return floor + (base_lr - floor) * (1.0 - progress)
    # cosine
    return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class LearningRateSchedule:
    """Applies `lr_at_step` to an optimizer's parameter groups.

    Each group's own initial lr is scaled, so a group configured at a different
    rate keeps its relative position through the schedule.
    """

    def __init__(self, optimizer: object, config: SchedulerConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.base_lrs: list[float] = [
            group["lr"] for group in optimizer.param_groups  # type: ignore[attr-defined]
        ]
        self.last_lr: float = self.base_lrs[0] if self.base_lrs else 0.0

    def set_step(self, step: int) -> float:
        """Set the learning rate for `step`. Returns the first group's rate."""
        groups = self.optimizer.param_groups  # type: ignore[attr-defined]
        for group, base in zip(groups, self.base_lrs, strict=True):
            group["lr"] = lr_at_step(step, base, self.config)
        self.last_lr = groups[0]["lr"] if groups else 0.0
        return self.last_lr

    def state_dict(self) -> dict[str, object]:
        return {"base_lrs": list(self.base_lrs), "config": vars(self.config)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        base = state.get("base_lrs")
        if isinstance(base, list) and len(base) == len(self.base_lrs):
            self.base_lrs = [float(x) for x in base]
