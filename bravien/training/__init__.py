"""Bravien training: optimizer, schedule, checkpointing, and the loop (§21–§26)."""

from __future__ import annotations

from bravien.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    CheckpointMetadata,
    LoadedCheckpoint,
    checkpoint_dir_for,
    load_checkpoint,
    prune_checkpoints,
    resolve_latest,
    restore_training_state,
    save_checkpoint,
    write_latest_pointer,
)
from bravien.training.optimizer import (
    OptimizerConfig,
    build_optimizer,
    parameter_group_summary,
    split_parameters,
)
from bravien.training.scheduler import (
    LearningRateSchedule,
    SchedulerConfig,
    lr_at_step,
)
from bravien.training.sft import (
    DEFAULT_SFT_LR,
    SFTRun,
    build_sft_dataset,
    default_sft_training_config,
    load_instruction_data,
    resolve_base_checkpoint,
    run_sft,
)
from bravien.training.trainer import (
    StepMetrics,
    Trainer,
    TrainingConfig,
    resolve_precision,
)

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "DEFAULT_SFT_LR",
    "CheckpointError",
    "CheckpointMetadata",
    "LearningRateSchedule",
    "LoadedCheckpoint",
    "OptimizerConfig",
    "SFTRun",
    "SchedulerConfig",
    "StepMetrics",
    "Trainer",
    "TrainingConfig",
    "build_optimizer",
    "build_sft_dataset",
    "checkpoint_dir_for",
    "default_sft_training_config",
    "load_checkpoint",
    "load_instruction_data",
    "lr_at_step",
    "parameter_group_summary",
    "prune_checkpoints",
    "resolve_base_checkpoint",
    "resolve_latest",
    "resolve_precision",
    "restore_training_state",
    "run_sft",
    "save_checkpoint",
    "split_parameters",
    "write_latest_pointer",
]
