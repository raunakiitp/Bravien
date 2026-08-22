"""Pretrain a Bravien model.

Trains Bravien's own transformer from random initialisation on a tokenized corpus
and writes checkpoints that `serve.py` can load. Nothing is downloaded and no
pretrained weights are adapted — the parameters that come out of this are the
ones this script created.

    python scripts/pretrain.py --preset tiny --max-steps 2000
    python scripts/pretrain.py --preset small --batch-size 4 --grad-accum 8
    python scripts/pretrain.py --resume            # continue the latest checkpoint
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bravien.utils.logging import configure_stdout, setup_logging

#: The scripts expose torch's dtype spelling; the trainer uses short names. Same
#: mapping as scripts/serve.py, for the same reason.
PRECISION_NAMES = {
    "auto": "auto",
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain a Bravien model on a tokenized corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--tokens",
        default=None,
        help="Packed token file (.bin) from prepare_data.py. "
        "Defaults to datasets/prepared/<name>.bin.",
    )
    data.add_argument("--dataset-name", default="bravien-corpus")
    data.add_argument("--dataset-dir", default="datasets/prepared")
    data.add_argument("--tokenizer", default="tokenizers/bravien")

    model = parser.add_argument_group("model")
    model.add_argument(
        "--preset",
        default="tiny",
        choices=["tiny", "small", "base"],
        help="Architecture size. Check VRAM before choosing base.",
    )
    model.add_argument("--hidden-size", type=int, default=None)
    model.add_argument("--num-layers", type=int, default=None)
    model.add_argument("--num-heads", type=int, default=None)
    model.add_argument("--num-kv-heads", type=int, default=None)
    model.add_argument(
        "--max-position-embeddings",
        type=int,
        default=None,
        help="Context length. Must be > --seq-len.",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--run-name", default="bravien")
    training.add_argument("--output-dir", default="checkpoints")
    training.add_argument("--max-steps", type=int, default=1000)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Optimizer steps see batch-size x grad-accum sequences.",
    )
    training.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Training sequence length. Defaults to the context length minus one.",
    )
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--warmup-steps", type=int, default=None)
    training.add_argument("--min-lr-ratio", type=float, default=0.1)
    training.add_argument("--max-grad-norm", type=float, default=1.0)
    training.add_argument(
        "--precision",
        default="auto",
        choices=sorted(PRECISION_NAMES),
    )
    training.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"])
    training.add_argument("--log-every", type=int, default=10)
    training.add_argument("--eval-every", type=int, default=200)
    training.add_argument("--eval-batches", type=int, default=20)
    training.add_argument("--save-every", type=int, default=500)
    training.add_argument("--keep-checkpoints", type=int, default=3)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--num-workers", type=int, default=0)
    training.add_argument(
        "--val-fraction",
        type=float,
        default=0.005,
        help="Tail of the corpus held out for evaluation.",
    )
    training.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the latest checkpoint in the run directory.",
    )
    training.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any existing checkpoint and start from random weights.",
    )
    training.add_argument(
        "--compile",
        action="store_true",
        dest="compile_model",
        help="torch.compile the model. Slow first step, faster afterwards.",
    )
    training.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level.upper())

    if args.resume and args.fresh:
        print("--resume and --fresh contradict each other.", file=sys.stderr)
        return 2

    from bravien.model.config import get_preset
    from bravien.training.optimizer import OptimizerConfig
    from bravien.training.pretrain import PretrainRun, run_pretraining
    from bravien.training.scheduler import SchedulerConfig
    from bravien.training.trainer import TrainingConfig

    token_path = (
        Path(args.tokens)
        if args.tokens
        else Path(args.dataset_dir) / f"{args.dataset_name}.bin"
    )
    if not token_path.exists():
        print(
            f"No token file at {token_path}\n"
            "Prepare a corpus first: python scripts/prepare_data.py",
            file=sys.stderr,
        )
        return 2

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(
            f"No tokenizer at {tokenizer_path}\n"
            "Train one first: python scripts/train_tokenizer.py",
            file=sys.stderr,
        )
        return 2

    model_config = get_preset(args.preset)
    overrides = {
        key: value
        for key, value in (
            ("hidden_size", args.hidden_size),
            ("num_layers", args.num_layers),
            ("num_heads", args.num_heads),
            ("num_kv_heads", args.num_kv_heads),
            ("max_position_embeddings", args.max_position_embeddings),
        )
        if value is not None
    }
    if overrides:
        # `replace` re-validates, so an inconsistent override (heads not dividing
        # hidden size, for instance) fails here rather than mid-training.
        model_config = model_config.replace(**overrides)

    # Warmup defaults to 10% of the run, which is more useful than a fixed 100
    # steps when max_steps is small.
    warmup = (
        args.warmup_steps
        if args.warmup_steps is not None
        else max(1, min(args.max_steps // 10, 2000))
    )

    training = TrainingConfig(
        run_name=args.run_name,
        stage="pretrain",
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
        compile_model=args.compile_model,
    )

    run = PretrainRun(
        token_path=token_path,
        tokenizer_path=tokenizer_path,
        model=model_config,
        training=training,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        resume=not args.fresh,
    )

    summary = run_pretraining(run)

    print()
    print("Training finished.")
    for key in (
        "steps",
        "tokens_seen",
        "epochs",
        "parameters",
        "first_loss",
        "final_loss",
        "best_eval_loss",
        "tokens_per_second",
        "status",
    ):
        value = summary.get(key)
        if value is not None:
            print(f"  {key:<18} {_format(value)}")

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
