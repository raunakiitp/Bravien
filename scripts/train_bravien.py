"""CLI entrypoint to fine-tune Qwen2.5 on Stage 2 data and produce checkpoints/bravien-v1.

Usage:
    python scripts/train_bravien.py
    python scripts/train_bravien.py --max-steps 300 --batch-size 2 --grad-accum 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bravien.training.hf_trainer import HFTrainer, HFTrainingConfig
from bravien.utils.logging import configure_stdout, get_logger, setup_logging

logger = get_logger("scripts.train_bravien")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen/Qwen2.5-0.5B-Instruct into Bravien-v1 using Stage 2 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base pretrained HuggingFace model or path",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/bravien-v1",
        help="Directory to save final Bravien model and checkpoints",
    )
    parser.add_argument(
        "--train-data",
        default="data/processed/train.jsonl",
        help="Path to Stage 2 train.jsonl",
    )
    parser.add_argument(
        "--val-data",
        default="data/processed/val.jsonl",
        help="Path to Stage 2 val.jsonl",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Total optimizer steps to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Micro-batch size per forward step",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Number of gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--precision",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Compute precision for model forward/backward",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum conversation sequence length in tokens",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Logging interval in optimizer steps",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Validation evaluation interval in optimizer steps",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Checkpoint save interval in optimizer steps",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    setup_logging(level="INFO")
    args = build_parser().parse_args(argv)

    config = HFTrainingConfig(
        base_model=args.base_model,
        output_dir=Path(args.output_dir),
        train_data_path=Path(args.train_data),
        val_data_path=Path(args.val_data),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        precision=args.precision,
        max_seq_length=args.max_seq_length,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )

    trainer = HFTrainer(config)
    result = trainer.train()

    print("\n=======================================================")
    print("BRAVIEN STAGE 3 SFT TRAINING COMPLETE")
    print("=======================================================")
    print(f"Final Checkpoint:     {result['checkpoint_path']}")
    print(f"Total Steps:          {result['steps']}")
    print(f"Best Validation Loss: {result['best_val_loss']:.4f}")
    print(f"Total Tokens Seen:    {result['total_tokens_seen']:,}")
    print(f"Elapsed Time:         {result['elapsed_seconds']:.1f}s")
    print("=======================================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
