"""Pretraining entry point (§23).

Wires the prepared corpus, the trained tokenizer and a model config into a run.
The one rule enforced here that cannot be enforced anywhere else: the model's
vocabulary is taken from the tokenizer, never from a config file. A mismatch
produces a model that emits ids the tokenizer cannot decode, and it is invisible
until generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bravien.data.dataset import (
    PretrainDataset,
    contiguous_splits,
    make_pretrain_dataloader,
)
from bravien.model.config import BravienConfig
from bravien.model.model import BravienForCausalLM
from bravien.tokenizer.tokenizer import BravienTokenizer
from bravien.training.checkpoint import load_checkpoint, resolve_latest
from bravien.training.trainer import Trainer, TrainingConfig
from bravien.utils.logging import format_count, get_logger
from bravien.utils.seeding import seed_everything

logger = get_logger("training.pretrain")


@dataclass
class PretrainRun:
    """Inputs for one pretraining run."""

    token_path: Path
    tokenizer_path: Path
    model: BravienConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)

    batch_size: int = 8
    seq_len: int | None = None
    num_workers: int = 0
    val_fraction: float = 0.005
    min_val_tokens: int = 4096
    #: Resume from the newest checkpoint in the run directory if one exists.
    resume: bool = True

    def __post_init__(self) -> None:
        self.token_path = Path(self.token_path)
        self.tokenizer_path = Path(self.tokenizer_path)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


def build_model(config: BravienConfig, tokenizer: BravienTokenizer) -> BravienForCausalLM:
    """Instantiate a model whose vocabulary is the tokenizer's, by construction."""
    if config.vocab_size != tokenizer.vocab_size:
        logger.info(
            "setting vocab_size from the tokenizer: %d -> %d",
            config.vocab_size,
            tokenizer.vocab_size,
        )
        config = config.replace(vocab_size=tokenizer.vocab_size)
    if config.pad_token_id != tokenizer.pad_token_id:
        config = config.replace(pad_token_id=tokenizer.pad_token_id)
    if config.eos_token_id != tokenizer.eos_token_id:
        config = config.replace(eos_token_id=tokenizer.eos_token_id)
    if config.bos_token_id != tokenizer.bos_token_id:
        config = config.replace(bos_token_id=tokenizer.bos_token_id)
    return BravienForCausalLM(config)


def run_pretraining(run: PretrainRun) -> dict[str, Any]:
    """Execute a pretraining run and return its summary."""
    seed_everything(run.training.seed)

    tokenizer = BravienTokenizer.from_pretrained(run.tokenizer_path)
    model = build_model(run.model, tokenizer)

    # A dataset item is seq_len + 1 tokens: the model shifts internally, so the
    # extra token is the target for the final position. The *block* the model
    # sees is therefore seq_len + 1 positions, and that is what has to fit inside
    # the trained context. Defaulting to max - 1 makes a block exactly fill it.
    max_ctx = model.config.max_position_embeddings
    seq_len = run.seq_len or max_ctx - 1

    if seq_len + 1 > max_ctx:
        raise ValueError(
            f"seq_len {seq_len} needs {seq_len + 1} positions (one extra for the "
            f"final target) but the model's max_position_embeddings is {max_ctx}; "
            f"use seq_len={max_ctx - 1} or a longer context"
        )

    probe = PretrainDataset(run.token_path, seq_len)
    total_tokens = int(probe.info["tokens"])
    corpus_checksum = probe.info.get("checksum")
    corpus_tokenizer_checksum = probe.info.get("tokenizer_checksum")

    if corpus_tokenizer_checksum and corpus_tokenizer_checksum != tokenizer.metadata.get(
        "vocab_checksum"
    ):
        raise ValueError(
            "this corpus was tokenized with a different tokenizer "
            f"(corpus {corpus_tokenizer_checksum[:12]}..., "
            f"current {str(tokenizer.metadata.get('vocab_checksum'))[:12]}...). "
            "Re-run data preparation or point at the matching tokenizer."
        )

    train_bounds, val_bounds = contiguous_splits(
        total_tokens,
        val_fraction=run.val_fraction,
        min_val_tokens=min(run.min_val_tokens, max(total_tokens // 20, seq_len + 1)),
    )
    train_set = PretrainDataset(run.token_path, seq_len, bounds=train_bounds)
    val_set = PretrainDataset(run.token_path, seq_len, bounds=val_bounds)

    train_loader = make_pretrain_dataloader(
        train_set,
        batch_size=run.batch_size,
        num_workers=run.num_workers,
        seed=run.training.seed,
    )
    val_loader = make_pretrain_dataloader(
        val_set,
        batch_size=run.batch_size,
        shuffle=False,
        num_workers=0,
        seed=run.training.seed,
        drop_last=False,
    )

    dataset_info = {
        "token_path": str(run.token_path),
        "tokens": total_tokens,
        "checksum": corpus_checksum,
        "seq_len": seq_len,
        "train_sequences": len(train_set),
        "validation_sequences": len(val_set),
        "documents": probe.info.get("documents"),
        "sources": list((probe.info.get("tokens_per_source") or {}).keys()),
    }

    trainer = Trainer(
        model,
        run.training,
        train_loader,
        eval_loader=val_loader,
        tokenizer=tokenizer,
        dataset_info=dataset_info,
    )

    if run.resume:
        latest = resolve_latest(run.training.run_dir)
        if latest is not None:
            logger.info("resuming from %s", latest)
            loaded = load_checkpoint(
                latest,
                device=trainer.device,
                load_tokenizer=False,
                load_training_state=True,
            )
            target = getattr(trainer.model, "_orig_mod", trainer.model)
            target.load_state_dict(loaded.model.state_dict(), strict=False)
            if loaded.training_state:
                trainer.load_training_state(loaded.training_state)

    report = model.parameter_report()
    tokens_per_step = run.batch_size * seq_len * run.training.grad_accum_steps
    logger.info(
        "%s: %s params, %s tokens/step, %s train sequences",
        model.config.name,
        format_count(report.total),
        format_count(tokens_per_step),
        format_count(len(train_set)),
    )
    epochs = tokens_per_step * run.training.max_steps / max(total_tokens, 1)
    logger.info(
        "run covers %s tokens = %.2f epoch(s) over the corpus",
        format_count(tokens_per_step * run.training.max_steps),
        epochs,
    )
    if epochs > 20:
        # Not an error: a tiny corpus is a legitimate smoke-test setup. But the
        # resulting loss curve measures memorisation, and saying so is the honest
        # thing (§71).
        logger.warning(
            "%.1f epochs over this corpus - falling loss will largely reflect "
            "memorisation, not language modelling",
            epochs,
        )

    summary = trainer.train()
    summary["parameters"] = report.total
    summary["dataset"] = dataset_info
    summary["epochs"] = round(epochs, 3)
    return summary
