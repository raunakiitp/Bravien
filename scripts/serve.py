"""Serve a Bravien checkpoint on localhost.

This is the process the web app talks to. It loads Bravien's own weights and
answers from them; nothing here reaches the network for inference, so once a
checkpoint exists the machine can be offline.

    python scripts/serve.py
    python scripts/serve.py --checkpoint checkpoints/bravien/step-0001000
    python scripts/serve.py --port 8100 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bravien.utils.logging import configure_stdout, setup_logging

#: The scripts expose torch's dtype spelling; the engine uses short names.
DTYPE_NAMES = {
    "auto": "auto",
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a Bravien checkpoint or pretrained model over a local HTTP API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint directory, run directory, or model repo ID (e.g. HuggingFaceTB/SmolLM-135M-Instruct). "
            "Falls back to $BRAVIEN_CHECKPOINT, $BRAVIEN_MODEL, then checkpoints/bravien."
        ),
    )
    parser.add_argument(
        "--model",
        "--hf-model",
        dest="model",
        default=None,
        help="Pretrained Hugging Face model identifier (e.g. HuggingFaceTB/SmolLM-135M-Instruct or Qwen/Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Loopback by default: the API has no authentication.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Override device selection (default: best available).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Weight precision. 'auto' picks the best for the device.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Concurrent generations. Above 1 they contend for the same weights.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Default response length when a request does not specify one.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="1.0 disables it. Small checkpoints loop without some penalty.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level.upper())

    from bravien.inference.engine import EngineConfig
    from bravien.inference.server import resolve_checkpoint_path, serve
    from bravien.model.generation import GenerationConfig

    target = args.model or args.checkpoint
    checkpoint = resolve_checkpoint_path(target)

    # If it's a local path and doesn't exist, check fallback
    if isinstance(checkpoint, Path) and not checkpoint.exists() and not args.model:
        default_hf = "HuggingFaceTB/SmolLM-135M-Instruct"
        print(
            f"Note: Local checkpoint not found at {checkpoint}.\n"
            f"Loading smart pretrained open weights: {default_hf}...\n",
            file=sys.stderr,
        )
        checkpoint = default_hf

    engine_config = EngineConfig(
        checkpoint=checkpoint if isinstance(checkpoint, Path) else None,
        device=args.device,
        dtype=DTYPE_NAMES[args.dtype],
        default_generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
            top_p=args.top_p if args.top_p < 1.0 else None,
            repetition_penalty=args.repetition_penalty,
        ),
    )

    serve(
        checkpoint,
        host=args.host,
        port=args.port,
        engine_config=engine_config,
        max_concurrency=max(1, args.max_concurrency),
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
