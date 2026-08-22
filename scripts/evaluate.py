"""Evaluate a Bravien checkpoint and write a report.

Runs every measurement in `bravien.evaluation` against a checkpoint's own weights
and tokenizer, then writes a JSON report and a markdown summary. Nothing here
reaches the network, and nothing compares Bravien to a hosted model — the report
states what was measured, against what baseline, and what was not measured at all
(§27, §71, §72).

    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/bravien-sft
    python scripts/evaluate.py --suite structure --suite identity
    python scripts/evaluate.py --texts datasets/heldout.jsonl --output reports/eval.json

With no `--texts`, perplexity is measured on a held-out slice of the synthetic
seed corpus and the report labels it synthetic. Pass real held-out text for a
number that says more.
"""

from __future__ import annotations

import argparse
import json
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
        description="Measure a Bravien checkpoint and write an evaluation report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint directory, or a run directory whose latest checkpoint should "
            "be used. Falls back to $BRAVIEN_CHECKPOINT, then checkpoints/bravien."
        ),
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        metavar="NAME",
        help="Suite to run; repeatable. Default: every built-in suite.",
    )
    parser.add_argument(
        "--texts",
        default=None,
        help=(
            "Held-out corpus for perplexity: a .jsonl file with a 'text' field, or a "
            "plain-text file split on blank lines. Default: synthetic seed corpus."
        ),
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=64,
        help="How many held-out documents to score for perplexity.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Cap for the behavioural generations.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the JSON report. Default: <checkpoint>/evaluation.json.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Also write a markdown summary here. Default: alongside the JSON.",
    )
    parser.add_argument(
        "--no-behaviour",
        action="store_true",
        help="Skip the generation-shape measurements (the slowest part).",
    )
    parser.add_argument(
        "--no-safety",
        action="store_true",
        help="Skip the refusal probes. The template-integrity check always runs.",
    )
    parser.add_argument(
        "--no-memorisation",
        action="store_true",
        help="Skip the verbatim-overlap measurement.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Override device selection (default: best available).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["auto", "float32", "float16", "bfloat16"],
        help=(
            "Weight precision. Full precision by default: a reported number should "
            "not depend on the accumulation order of a reduced-precision kernel."
        ),
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="Print the available suites and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    return parser


def load_texts(path: Path, limit: int) -> list[str]:
    """Read held-out documents from JSONL or plain text.

    A malformed JSONL line is skipped rather than fatal — an evaluation corpus is
    input data, and one bad line should not lose the run.
    """
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")

    texts: list[str] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text") if isinstance(record, dict) else None
                if isinstance(text, str) and text.strip():
                    texts.append(text)
                if len(texts) >= limit:
                    break
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
        texts = [block.strip() for block in raw.split("\n\n") if block.strip()][:limit]

    if not texts:
        raise ValueError(f"{path} contained no usable documents")
    return texts


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level.upper())

    # Imported after argument parsing so `--help` does not pay for loading torch.
    from bravien.evaluation import all_suite_names
    from bravien.evaluation.evaluator import EvaluationConfig, evaluate_checkpoint

    if args.list_suites:
        from bravien.evaluation import SUITES

        for name in all_suite_names():
            suite = SUITES[name]
            print(f"{name:<12} {suite.kind:<16} {suite.description}")
        return 0

    from bravien.inference.engine import EngineConfig, InferenceEngine
    from bravien.inference.server import resolve_checkpoint_path
    from bravien.model.generation import GenerationConfig

    checkpoint = resolve_checkpoint_path(args.checkpoint)
    if not checkpoint.exists():
        print(
            f"No checkpoint at {checkpoint}\n\n"
            "Train one first:\n"
            "  python scripts/train_tokenizer.py\n"
            "  python scripts/prepare_data.py\n"
            "  python scripts/pretrain.py\n\n"
            "Or point at an existing one with --checkpoint / $BRAVIEN_CHECKPOINT.",
            file=sys.stderr,
        )
        return 2

    unknown = sorted(set(args.suite or ()) - set(all_suite_names()))
    if unknown:
        print(
            f"Unknown suite(s): {', '.join(unknown)}\n"
            f"Available: {', '.join(all_suite_names())}",
            file=sys.stderr,
        )
        return 2

    texts = None
    if args.texts:
        try:
            texts = load_texts(Path(args.texts), args.documents)
        except (FileNotFoundError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2

    engine = InferenceEngine.from_checkpoint(
        checkpoint,
        config=EngineConfig(
            device=args.device,
            dtype=DTYPE_NAMES[args.dtype],
            # The suites set their own caps per item; this is only the ceiling a
            # request cannot exceed, so it has to be at least the largest of them.
            max_new_tokens_limit=max(args.max_new_tokens, 128),
            default_generation=GenerationConfig(max_new_tokens=args.max_new_tokens),
        ),
    )

    report = evaluate_checkpoint(
        engine,
        texts=texts,
        config=EvaluationConfig(
            suites=tuple(args.suite) if args.suite else (),
            perplexity_documents=args.documents,
            behaviour_max_new_tokens=args.max_new_tokens,
            include_behaviour=not args.no_behaviour,
            include_safety=not args.no_safety,
            include_memorisation=not args.no_memorisation,
        ),
    )

    json_path = Path(args.output) if args.output else checkpoint / "evaluation.json"
    report.save(json_path)

    markdown_path = (
        Path(args.markdown) if args.markdown else json_path.with_suffix(".md")
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")

    print()
    print(report.to_markdown())
    print(f"JSON:     {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
