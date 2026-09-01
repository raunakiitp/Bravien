"""Rolling checkpoint manager and recovery handler for native Bravien pretraining."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from bravien.model.bravien_model import BravienForCausalLM
from bravien.model.checkpoint import save_bravien_checkpoint


@dataclass
class TrainingState:
    step: int
    epoch: int
    best_loss: float
    total_tokens_trained: int
    elapsed_seconds: float
    optimizer_state_dict: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "total_tokens_trained": self.total_tokens_trained,
            "elapsed_seconds": self.elapsed_seconds,
        }


class CheckpointManager:
    """Manages rolling checkpoints, step directories, and atomic recovery for training."""

    def __init__(self, output_dir: str | Path, keep_last_n: int = 3) -> None:
        self.output_dir = Path(output_dir)
        self.keep_last_n = keep_last_n
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: list[tuple[int, Path]] = []
        self._scan_existing_checkpoints()

    def _scan_existing_checkpoints(self) -> None:
        for p in self.output_dir.iterdir():
            if p.is_dir() and p.name.startswith("step_"):
                try:
                    step = int(p.name.split("_")[1])
                    self._checkpoints.append((step, p))
                except ValueError:
                    pass
        self._checkpoints.sort(key=lambda x: x[0])

    def save_checkpoint(
        self,
        model: BravienForCausalLM,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        state: TrainingState,
        is_best: bool = False,
    ) -> Path:
        """Atomically save model weights, optimizer states, and training metadata."""
        step_dir = self.output_dir / f"step_{state.step:07d}"
        temp_dir = self.output_dir / f"temp_step_{state.step:07d}"

        # 1. Save standard model checkpoint
        save_bravien_checkpoint(
            model=model,
            save_directory=temp_dir,
            manifest_metadata={"training_step": state.step, "loss": state.best_loss},
            training_steps=state.step,
            training_loss=state.best_loss,
        )

        # 2. Save Optimizer & Scheduler states
        train_state_file = temp_dir / "training_state.pt"
        torch.save(
            {
                "training_state": state.to_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            },
            train_state_file,
        )

        # 3. Save JSON training summary
        summary_file = temp_dir / "training_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)

        # Atomic rename
        if step_dir.exists():
            shutil.rmtree(step_dir)
        temp_dir.rename(step_dir)

        # Track rolling checkpoints
        self._checkpoints.append((state.step, step_dir))
        self._checkpoints.sort(key=lambda x: x[0])

        # Remove older checkpoints beyond keep_last_n
        while len(self._checkpoints) > self.keep_last_n:
            oldest_step, oldest_path = self._checkpoints.pop(0)
            if oldest_path.exists():
                shutil.rmtree(oldest_path, ignore_errors=True)

        # Save symlink/copy for best model if indicated
        if is_best:
            best_dir = self.output_dir / "best_model"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(step_dir, best_dir)

        return step_dir

    def get_latest_checkpoint(self) -> Path | None:
        if not self._checkpoints:
            return None
        return self._checkpoints[-1][1]
