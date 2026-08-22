"""Train Bravien's tokenizer.

The tokenizer is Bravien's own: byte-level BPE trained on the corpus, with
reserved ids for the special tokens the chat template and training code rely on.
It is not borrowed from another model, and every checkpoint carries a copy so a
saved model can always decode its own output.

    python scripts/train_tokenizer.py --vocab-size 32000
    python scripts/train_tokenizer.py --input datasets/raw --vocab-size 16000
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bravien.utils.logging import configure_stdout, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Bravien byte-level BPE tokenizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Directory or file of training text. Omit to use the synthetic seed "
            "corpus, which is fine for a smoke run and useless for a real model."
        ),
    )
    parser.add_argument("--output", default="tokenizers/bravien")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Merges seen fewer times than this are discarded.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Cap the documents read, for a faster pass over a huge corpus.",
    )
    parser.add_argument(
        "--synthetic-documents",
        type=int,
        default=4000,
        help="Size of the synthetic corpus when --input is omitted.",
    )
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level.upper())

    from bravien.data.prepare import generate_seed_corpus, load_directory, load_text_file
    from bravien.tokenizer.train import summarize, train_tokenizer

    if args.input:
        source = Path(args.input)
        if not source.exists():
            print(f"No such input: {source}", file=sys.stderr)
            return 2
        documents = (
            load_directory(source) if source.is_dir() else load_text_file(source)
        )
        origin = str(source)
    else:
        print(
            "No --input given: training on the SYNTHETIC seed corpus.\n"
            "A tokenizer trained on synthetic text will tokenise real text poorly. "
            "Use it to verify the pipeline, not to train a model you intend to use.",
            file=sys.stderr,
        )
        documents = generate_seed_corpus(args.synthetic_documents)
        origin = "synthetic-seed-corpus"

    def texts() -> Iterator[str]:
        count = 0
        for document in documents:
            text = getattr(document, "text", document)
            if not isinstance(text, str) or not text.strip():
                continue
            yield text
            count += 1
            if args.max_documents and count >= args.max_documents:
                break

    output = Path(args.output)
    tokenizer = train_tokenizer(
        texts(),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=output,
        sources=[{"origin": origin, "synthetic": args.input is None}],
    )

    info = summarize(tokenizer)
    print()
    print(info)
    print()
    print(f"Written to {output.resolve()}")
    print("Next: python scripts/prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
