"""Reproducible seeding (§23).

Same seed plus same data order must give the same loss curve, or nothing else in
the training pipeline can be debugged.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SeedState:
    """Everything needed to resume a run at the same point in every RNG stream."""

    seed: int
    python_state: tuple
    numpy_state: dict
    torch_state: torch.Tensor
    cuda_states: list[torch.Tensor] | None = None


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and Torch.

    Args:
        deterministic: also force deterministic cuDNN/cuBLAS kernels. This costs
            real throughput, so it is opt-in: use it when chasing a bug, not for
            production runs.
    """
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must fit in 32 bits, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS needs this set before any handle is created to make matmuls
        # reproducible; setting it later has no effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def capture_seed_state(seed: int) -> SeedState:
    """Snapshot every RNG so a checkpoint can resume mid-epoch (§26)."""
    return SeedState(
        seed=seed,
        python_state=random.getstate(),
        numpy_state=np.random.get_state(legacy=False),
        torch_state=torch.get_rng_state(),
        cuda_states=(
            list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
        ),
    )


def restore_seed_state(state: SeedState) -> None:
    """Restore RNG streams captured by `capture_seed_state`."""
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_state.cpu().to(torch.uint8))
    if state.cuda_states and torch.cuda.is_available():
        available = torch.cuda.device_count()
        if len(state.cuda_states) == available:
            torch.cuda.set_rng_state_all(
                [s.cpu().to(torch.uint8) for s in state.cuda_states]
            )
        # A different GPU count than the one that saved the checkpoint cannot be
        # restored faithfully; the CPU streams above still are.


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, reproducible seed.

    Without this, forked workers share a seed and any per-worker randomness is
    duplicated across them.
    """
    base = torch.initial_seed() % 2**32
    seed = (base + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)
