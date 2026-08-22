"""Build a tokenized training corpus.

Cleans, filters, deduplicates and tokenizes documents into a packed token file
that `pretrain.py` reads. The manifest records where every document came from and
how many survived each stage, so a checkpoint's training data is auditable rather
than a claim.

    python scripts/prepare_data.py
    python scripts/prepare_data.py --local-dir datasets/raw/mytexts --name mycorpus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bravien.utils.logging import configure_stdout, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean, filter, deduplicate and tokenize a training corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizers/bravien",
        help="Tokenizer directory from train_tokenizer.py.",
    )
    parser.add_argument("--output-dir", default="datasets/prepared")
    parser.add_argument("--raw-dir", default="datasets/raw")
    parser.add_argument("--name", default="bravien-corpus")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Directory of .txt/.jsonl files to use instead of downloading.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1024,
        help="Packing length. Must be >= the seq_len used for training.",
    )
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="Fail instead of falling back to the synthetic corpus when no real "
        "documents are found.",
    )
    parser.add_argument("--synthetic-documents", type=int, default=4000)
    parser.add_argument(
        "--no-deduplicate", action="store_true", help="Skip exact + near dedup."
    )
    parser.add_argument(
        "--no-redact-pii",
        action="store_true",
        help="Keep e-mail addresses and phone numbers as-is.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level.upper())

    from bravien.data.prepare import PrepareConfig, prepare_dataset
    from bravien.tokenizer.tokenizer import BravienTokenizer

    tokenizer_dir = Path(args.tokenizer)
    if not tokenizer_dir.exists():
        print(
            f"No tokenizer at {tokenizer_dir}\n"
            "Train one first: python scripts/train_tokenizer.py",
            file=sys.stderr,
        )
        return 2

    tokenizer = BravienTokenizer.from_pretrained(tokenizer_dir)

    config = PrepareConfig(
        output_dir=Path(args.output_dir),
        raw_dir=Path(args.raw_dir),
        name=args.name,
        sequence_length=args.sequence_length,
        local_dir=Path(args.local_dir) if args.local_dir else None,
        allow_synthetic_fallback=not args.no_synthetic_fallback,
        synthetic_documents=args.synthetic_documents,
        deduplicate=not args.no_deduplicate,
        near_duplicates=not args.no_deduplicate,
        redact_pii=not args.no_redact_pii,
        seed=args.seed,
    )

    result = prepare_dataset(config, tokenizer)

    print()
    print(f"Tokens written to {result.token_path}")
    print(f"Manifest         {result.manifest_path}")
    print(f"  tokens         {result.tokens:,}")
    print(f"  sequences      {result.sequences:,} x {args.sequence_length}")

    sources = [s.name for s in result.manifest.sources]
    print(f"  sources        {', '.join(sources) if sources else 'none recorded'}")

    # The synthetic corpus is labelled at its source, so this reads the manifest
    # rather than re-deriving it from the flags.
    if any("synthetic" in name for name in sources):
        print()
        print(
            "This corpus is SYNTHETIC. A model trained on it learns the generator's "
            "patterns, not language. Point --local-dir at real text for a model you "
            "intend to use."
        )
    print()
    print("Next: python scripts/pretrain.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
