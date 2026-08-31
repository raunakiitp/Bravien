"""CLI command to build, clean, filter, and export the Bravien training dataset.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --output-dir data/processed --sample-dir data/samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bravien.data.pipeline import BravienDataPipeline, PipelineConfig
from bravien.utils.logging import configure_stdout, get_logger, setup_logging

logger = get_logger("scripts.prepare_dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, clean, filter, deduplicate, and split the Bravien training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save train.jsonl, val.jsonl, and test.jsonl",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/manifests",
        help="Directory to save dataset_manifest.json",
    )
    parser.add_argument(
        "--sample-dir",
        default="data/samples",
        help="Directory to save sample_conversations.jsonl",
    )
    parser.add_argument(
        "--raw-sources",
        nargs="*",
        default=["datasets/alpaca_data.jsonl"],
        help="Raw JSONL files to ingest",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Proportion of data assigned to training set",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.075,
        help="Proportion of data assigned to validation set",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.075,
        help="Proportion of data assigned to test set",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.6,
        help="Minimum quality score threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=50000,
        help="Target number of high-quality examples in final dataset",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        default=True,
        help="Enforce category balancing across all domains",
    )
    parser.add_argument(
        "--max-category-share",
        type=float,
        default=0.35,
        help="Maximum proportion allowed for any single category",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic hashing and splitting",
    )
    parser.add_argument(
        "--version",
        default="2.0.0",
        help="Dataset release version",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    setup_logging(level="INFO")
    args = build_parser().parse_args(argv)

    raw_paths = [Path(p) for p in args.raw_sources if Path(p).exists()]
    if not raw_paths:
        logger.warning(f"No configured raw source files found from {args.raw_sources}. Proceeding with curated seeds.")

    config = PipelineConfig(
        output_dir=Path(args.output_dir),
        manifest_dir=Path(args.manifest_dir),
        sample_dir=Path(args.sample_dir),
        raw_sources=raw_paths,
        include_curated_seeds=True,
        include_synthetic_scaling=True,
        target_samples=args.target_samples,
        balance_categories=args.balance,
        max_category_share=args.max_category_share,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_quality_score=args.min_quality,
        seed=args.seed,
        dataset_version=args.version,
    )

    pipeline = BravienDataPipeline(config)
    manifest = pipeline.run()

    print("\n=======================================================")
    print("BRAVIEN DATASET CREATION SUMMARY")
    print("=======================================================")
    print(f"Dataset Version:    {manifest['version']}")
    print(f"Total Ingested:     {manifest['statistics']['total_ingested']:,}")
    print(f"Total Kept:         {manifest['statistics']['total_kept']:,}")
    print(f"  - Train split:    {manifest['statistics']['train_samples']:,} ({args.train_ratio*100:.1f}%)")
    print(f"  - Val split:      {manifest['statistics']['val_samples']:,} ({args.val_ratio*100:.1f}%)")
    print(f"  - Test split:     {manifest['statistics']['test_samples']:,} ({args.test_ratio*100:.1f}%)")
    print(f"Manifest Checksum:  {manifest['manifest_sha256']}")
    print("=======================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
