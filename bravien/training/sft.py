"""Supervised fine-tuning (§26).

Instruction tuning is the same loop as pretraining with three differences, all of
which live here rather than in `Trainer`:

1. **Weights start from a base checkpoint.** This is an initialisation, not a
   resume: the step counter restarts at zero, the optimizer state is fresh, and
   the learning-rate schedule warms up again. Carrying a pretraining optimizer
   into SFT would apply momentum accumulated on a different objective.
2. **Supervision is masked to assistant spans**, which `build_sft_example`
   already does via the chat template's `trainable` flags.
3. **The tokenizer comes from the base checkpoint**, never from a path the caller
   guessed. A fine-tune whose tokenizer disagrees with its base model trains on
   ids that mean something else, and the loss curve looks fine while it happens.

The learning rate defaults an order of magnitude below pretraining's. Fine-tuning
at pretraining rates on a few thousand examples destroys the base model's
distribution faster than it teaches the format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bravien.data.dataset import SFTDataset, make_sft_dataloader
from bravien.data.instructions import (
    Conversation,
    InstructionDataError,
    LoadStats,
    generate_seed_conversations,
    jsonl_record,
    load_conversations_jsonl,
    seed_instructions_record,
    split_conversations,
)
from bravien.data.pack import SFTPackStats, build_sft_examples
from bravien.tokenizer.tokenizer import BravienTokenizer
from bravien.training.checkpoint import (
    CheckpointError,
    load_checkpoint,
    resolve_latest,
)
from bravien.training.optimizer import OptimizerConfig
from bravien.training.trainer import Trainer, TrainingConfig
from bravien.utils.logging import format_count, get_logger
from bravien.utils.seeding import seed_everything

logger = get_logger("training.sft")

#: Fine-tuning learning rate. Low by default for the reason in the module
#: docstring; override it deliberately, not by accident.
DEFAULT_SFT_LR = 1e-4


@dataclass
class SFTRun:
    """Inputs for one instruction-tuning run."""

    #: Base checkpoint directory, or a run directory whose newest checkpoint is
    #: used. Required: there is nothing to fine-tune without one.
    base_checkpoint: Path
    training: TrainingConfig = field(default_factory=TrainingConfig)

    #: JSONL of conversations. When None, the generated seed set is used and the
    #: run is labelled synthetic.
    data_path: Path | None = None
    seed_examples: int = 2000

    batch_size: int = 4
    #: Longest conversation in tokens. Defaults to the model's trained context.
    max_length: int | None = None
    num_workers: int = 0
    val_fraction: float = 0.05
    limit: int | None = None

    def __post_init__(self) -> None:
        self.base_checkpoint = Path(self.base_checkpoint)
        if self.data_path is not None:
            self.data_path = Path(self.data_path)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.training.stage == "pretrain":
            # The stage is written into checkpoint metadata and surfaced in the UI
            # badge; leaving it at the default would mislabel the result.
            self.training.stage = "sft"


def resolve_base_checkpoint(path: str | Path) -> Path:
    """Accept either a checkpoint directory or a run directory.

    Pointing at `checkpoints/bravien` and getting "not a checkpoint directory" is
    an unhelpful failure when the newest checkpoint is one level down and is
    obviously what was meant.
    """
    path = Path(path)
    if (path / "config.json").is_file():
        return path

    latest = resolve_latest(path)
    if latest is not None:
        logger.info("using newest checkpoint in %s: %s", path, latest.name)
        return latest

    raise CheckpointError(
        f"{path} is neither a checkpoint directory nor a run directory with a "
        "checkpoint in it. Pretrain first: python scripts/pretrain.py"
    )


def load_instruction_data(run: SFTRun) -> tuple[list[Conversation], dict[str, Any]]:
    """Read or generate conversations, plus provenance for the checkpoint."""
    if run.data_path is not None:
        stats = LoadStats()
        conversations = load_conversations_jsonl(
            run.data_path, stats=stats, limit=run.limit
        )
        record = jsonl_record(run.data_path, len(conversations))
        info = {
            "source": "jsonl",
            "path": str(run.data_path),
            "conversations": len(conversations),
            "load_stats": {
                "lines": stats.lines,
                "kept": stats.kept,
                "dropped": stats.dropped,
            },
            "synthetic": False,
            "provenance": asdict(record),
        }
        return conversations, info

    count = run.limit or run.seed_examples
    conversations = list(generate_seed_conversations(count, seed=run.training.seed))
    record = seed_instructions_record(len(conversations))
    logger.warning(
        "no --data given: fine-tuning on %s generated conversations. This teaches "
        "reply format, not knowledge — the result is a pipeline proof, not an "
        "assistant.",
        format_count(len(conversations)),
    )
    return conversations, {
        "source": "bravien-seed-instructions",
        "conversations": len(conversations),
        "synthetic": True,
        "provenance": asdict(record),
    }


def build_sft_dataset(
    conversations: list[Conversation],
    tokenizer: BravienTokenizer,
    *,
    max_length: int,
    source: str,
) -> tuple[SFTDataset, SFTPackStats]:
    """Tokenize conversations into a dataset, reporting what was lost."""
    stats = SFTPackStats()
    examples = list(
        build_sft_examples(
            conversations, tokenizer, max_length=max_length, source=source, stats=stats
        )
    )
    if not examples:
        raise InstructionDataError(
            f"none of the {stats.conversations:,} conversations produced a "
            f"trainable example at max_length={max_length}. Every one was either "
            "empty or had its assistant turn truncated away."
        )
    return SFTDataset(examples), stats


def run_sft(run: SFTRun) -> dict[str, Any]:
    """Execute an instruction-tuning run and return its summary."""
    seed_everything(run.training.seed)

    base_dir = resolve_base_checkpoint(run.base_checkpoint)

    # The tokenizer travels with the checkpoint, so vocabulary agreement is
    # structural rather than something the caller has to get right.
    loaded = load_checkpoint(base_dir, device="cpu", load_tokenizer=True)
    if loaded.tokenizer is None:
        raise CheckpointError(
            f"{base_dir} has no bundled tokenizer, so the ids it was trained on "
            "cannot be reproduced. Re-run pretraining with the tokenizer passed "
            "to the Trainer."
        )
    tokenizer: BravienTokenizer = loaded.tokenizer
    model = loaded.model

    max_ctx = model.config.max_position_embeddings
    max_length = run.max_length or max_ctx
    if max_length > max_ctx:
        raise ValueError(
            f"max_length={max_length} exceeds the model's trained context "
            f"({max_ctx}); RoPE would extrapolate into positions it has never seen"
        )

    conversations, data_info = load_instruction_data(run)
    split = split_conversations(
        conversations, val_fraction=run.val_fraction, seed=run.training.seed
    )
    logger.info("%s", split.describe())

    train_set, train_stats = build_sft_dataset(
        split.train, tokenizer, max_length=max_length, source=data_info["source"]
    )
    logger.info("train %s", train_set.describe())
    if train_stats.dropped_unsupervised:
        logger.warning(
            "%d conversation(s) dropped: no assistant tokens survived truncation",
            train_stats.dropped_unsupervised,
        )
    if train_stats.truncated:
        logger.info(
            "%d conversation(s) hit max_length=%d and lost their earliest context",
            train_stats.truncated,
            max_length,
        )

    train_loader = make_sft_dataloader(
        train_set,
        batch_size=run.batch_size,
        pad_id=tokenizer.pad_token_id,
        num_workers=run.num_workers,
        seed=run.training.seed,
    )

    val_loader = None
    if split.validation:
        val_set, _ = build_sft_dataset(
            split.validation,
            tokenizer,
            max_length=max_length,
            source=data_info["source"],
        )
        logger.info("validation %s", val_set.describe())
        val_loader = make_sft_dataloader(
            val_set,
            batch_size=run.batch_size,
            pad_id=tokenizer.pad_token_id,
            shuffle=False,
            num_workers=0,
            seed=run.training.seed,
        )

    dataset_info = {
        **data_info,
        "max_length": max_length,
        "train_conversations": len(train_set),
        "validation_conversations": len(split.validation),
        "supervised_tokens": train_set.supervised_tokens,
        "supervision_rate": round(train_stats.supervision_rate, 4),
        "base_checkpoint": str(base_dir),
        "base_step": loaded.metadata.step,
        "base_stage": loaded.metadata.stage,
    }

    trainer = Trainer(
        model,
        run.training,
        train_loader,
        eval_loader=val_loader,
        tokenizer=tokenizer,
        dataset_info=dataset_info,
    )

    # Deliberately after Trainer construction and deliberately *not* loading
    # `training_state`: the weights are inherited, the optimizer is not.
    logger.info(
        "initialised from %s (step %d, stage %s); step counter restarts at 0",
        base_dir.name,
        loaded.metadata.step,
        loaded.metadata.stage or "unknown",
    )

    report = model.parameter_report()
    logger.info(
        "fine-tuning %s (%s params) on %s supervised tokens, lr %.2e",
        model.config.name,
        format_count(report.total),
        format_count(train_set.supervised_tokens),
        run.training.optimizer.learning_rate,
    )

    epochs = (
        run.batch_size * run.training.grad_accum_steps * run.training.max_steps
    ) / max(len(train_set), 1)
    logger.info("run covers %.2f epoch(s) over the instruction set", epochs)
    if epochs > 5:
        logger.warning(
            "%.1f epochs of instruction tuning — past about 3 the model starts "
            "reciting the training replies rather than generalising",
            epochs,
        )

    summary = trainer.train()
    summary["parameters"] = report.total
    summary["dataset"] = dataset_info
    summary["epochs"] = round(epochs, 3)
    summary["base_checkpoint"] = str(base_dir)
    return summary


def default_sft_training_config(
    *,
    run_name: str = "bravien-sft",
    max_steps: int = 400,
    output_dir: Path | str = "checkpoints",
) -> TrainingConfig:
    """A TrainingConfig with fine-tuning defaults rather than pretraining ones."""
    return TrainingConfig(
        run_name=run_name,
        stage="sft",
        output_dir=Path(output_dir),
        max_steps=max_steps,
        optimizer=OptimizerConfig(learning_rate=DEFAULT_SFT_LR),
        eval_every=50,
        save_every=200,
        log_every=10,
    )
