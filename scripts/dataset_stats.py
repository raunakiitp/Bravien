"""Dataset statistics and distribution reporter for Bravien.

Summarizes:
1. Sample counts across splits
2. Category distributions
3. Language breakdown
4. Token length profiles (input, output, total)
5. Quality score metrics
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bravien.data.schema import BravienTrainingExample
from bravien.utils.logging import configure_stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate and display detailed metrics for a prepared Bravien dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/processed",
        help="Directory containing train.jsonl, val.jsonl, and test.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/dataset_manifest.json",
        help="Path to dataset_manifest.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    manifest_path = Path(args.manifest)

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return 1

    examples: list[BravienTrainingExample] = []
    split_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        p = dataset_dir / f"{split}.jsonl"
        if not p.exists():
            continue
        split_exs = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    split_exs.append(BravienTrainingExample.from_dict(json.loads(line)))
        split_counts[split] = len(split_exs)
        examples.extend(split_exs)

    if not examples:
        print("Error: No valid training examples found in dataset directory.")
        return 1

    total = len(examples)
    categories: dict[str, int] = {}
    languages: dict[str, int] = {}
    qualities = [ex.quality_score for ex in examples]
    token_counts = [ex.estimate_tokens() for ex in examples]

    for ex in examples:
        categories[ex.category] = categories.get(ex.category, 0) + 1
        languages[ex.language] = languages.get(ex.language, 0) + 1

    print("\n" + "=" * 60)
    print("BRAVIEN DATASET STATISTICAL REPORT")
    print("=" * 60)

    print("\n--- Split Breakdown ---")
    for s, count in split_counts.items():
        print(f"  {s.capitalize():10s}: {count:6,d} ({count / total * 100:5.1f}%)")
    print(f"  {'Total':10s}: {total:6,d} (100.0%)")

    print("\n--- Category Distribution ---")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:24s}: {count:6,d} ({count / total * 100:5.1f}%)")

    print("\n--- Language Distribution ---")
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang:24s}: {count:6,d} ({count / total * 100:5.1f}%)")

    print("\n--- Quality Score Metrics ---")
    print(f"  Mean Score       : {statistics.mean(qualities):.4f}")
    print(f"  Median Score     : {statistics.median(qualities):.4f}")
    print(f"  Min Score        : {min(qualities):.4f}")
    print(f"  Max Score        : {max(qualities):.4f}")

    print("\n--- Estimated Token Lengths ---")
    print(f"  Mean Tokens/Ex   : {statistics.mean(token_counts):.1f}")
    print(f"  Median Tokens    : {statistics.median(token_counts):.1f}")
    print(f"  Min Tokens       : {min(token_counts)}")
    print(f"  Max Tokens       : {max(token_counts)}")
    print(f"  Total Tokens Est : {sum(token_counts):,d}")

    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
