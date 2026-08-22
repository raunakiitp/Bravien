"""Instruction-tune a pretrained Bravien checkpoint.

Takes weights produced by `pretrain.py` and teaches them the conversation format:
that a user turn is a request, that the reply belongs to the assistant, and that
replies end. The output is a new checkpoint `serve.py` can load.

    python scripts/finetune.py                                  # generated data
    python scripts/finetune.py --data datasets/instructions.jsonl
    python scripts/finetune.py --base checkpoints/bravien/step-0001500 --max-steps 800

With no --data the run uses conversations generated in this repository. That
teaches reply *format* and a few facts about what Bravien is; it does not make the
model knowledgeable, and the checkpoint records the data as synthetic.

The instruction file is JSON Lines, one object per line. Either shape works:

    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    {"instruction": "...", "input": "...", "output": "..."}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bravien.utils.logging import configure_stdout, setup_logging

#: The scripts expose torch's dtype spelling; the trainer uses short names.
PRECISION_NAMES = {
    "auto": "auto",
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Instruction-tune a pretrained Bravien checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--base",
        default="checkpoints/bravien",
        help="Pretrained checkpoint, or a run directory whose newest checkpoint "
        "is used.",
    )
    data.add_argument(
        "--data",
        default=None,
        help="JSONL of conversations. Omit to use generated seed conversations.",
    )
    data.add_argument(
        "--seed-examples",
        type=int,
        default=2000,
        help="How many conversations to generate when --data is omitted.",
    )
    data.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use at most this many conversations. Useful for a quick trial run.",
    )
    data.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Longest conversation in tokens. Defaults to the model's context.",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--run-name", default="bravien-sft")
    training.add_argument("--output-dir", default="checkpoints")
    training.add_argument("--max-steps", type=int, default=400)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--grad-accum", type=int, default=1)
    training.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="An order of magnitude below pretraining on purpose.",
    )
    training.add_argument("--weight-decay", type=float, default=0.0)
    training.add_argument("--warmup-steps", type=int, default=None)
    training.add_argument("--min-lr-ratio", type=float, default=0.1)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument(
        "--precision", default="auto", choices=sorted(PRECISION_NAMES)
    )
    training.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"])
    training.add_argument("--log-every", type=int, default=10)
    training.add_argument("--eval-every", type=int, default=50)
    training.add_argument("--eval-batches", type=int, default=20)
    training.add_argument("--save-every", type=int, default=200)
    training.add_argument("--keep-checkpoints", type=int, default=3)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--num-workers", type=int, default=0)
    training.add_argument("--val-fraction", type=float, default=0.05)
    training.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level.upper())

    from bravien.data.instructions import InstructionDataError
    from bravien.training.checkpoint import CheckpointError
    from bravien.training.optimizer import OptimizerConfig
    from bravien.training.scheduler import SchedulerConfig
    from bravien.training.sft import SFTRun, run_sft
    from bravien.training.trainer import TrainingConfig

    base = Path(args.base)
    if not base.exists():
        print(
            f"No checkpoint at {base}\n"
            "Pretrain first: python scripts/pretrain.py",
            file=sys.stderr,
        )
        return 2

    if args.data is not None and not Path(args.data).is_file():
        print(f"No instruction file at {args.data}", file=sys.stderr)
        return 2

    if args.run_name == Path(args.base).name:
        # Writing the fine-tune into the run directory it was read from would let
        # checkpoint pruning delete the base weights mid-run.
        print(
            f"--run-name {args.run_name!r} collides with the base checkpoint's "
            "run directory; choose a different name so the base is not pruned.",
            file=sys.stderr,
        )
        return 2

    warmup = (
        args.warmup_steps
        if args.warmup_steps is not None
        else max(1, min(args.max_steps // 10, 200))
    )

    training = TrainingConfig(
        run_name=args.run_name,
        stage="sft",
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        optimizer=OptimizerConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        scheduler=SchedulerConfig(
            warmup_steps=warmup,
            total_steps=args.max_steps,
            min_lr_ratio=args.min_lr_ratio,
        ),
        precision=PRECISION_NAMES[args.precision],
        device=args.device,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        keep_checkpoints=args.keep_checkpoints,
        seed=args.seed,
    )

    run = SFTRun(
        base_checkpoint=base,
        training=training,
        data_path=Path(args.data) if args.data else None,
        seed_examples=args.seed_examples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        limit=args.limit,
    )

    try:
        summary = run_sft(run)
    except (CheckpointError, InstructionDataError) as error:
        print(f"\n{error}", file=sys.stderr)
        return 1

    print()
    print("Fine-tuning finished.")
    for key in (
        "steps",
        "tokens_seen",
        "epochs",
        "parameters",
        "first_loss",
        "final_loss",
        "best_eval_loss",
        "tokens_per_second",
        "stop_reason",
    ):
        value = summary.get(key)
        if value is not None:
            print(f"  {key:<18} {_format(value)}")

    dataset = summary.get("dataset", {})
    if dataset.get("synthetic"):
        print()
        print(
            "  This checkpoint was tuned on generated conversations. It has\n"
            "  learned a reply format, not knowledge. Use --data with a real\n"
            "  instruction set for a model worth talking to."
        )

    run_dir = Path(args.output_dir) / args.run_name
    print()
    print(f"Serve it: python scripts/serve.py --checkpoint {run_dir}")
    return 0


def _format(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
