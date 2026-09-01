"""CLI Entrypoint for Pretraining Native Bravien Models.

Supports:
- --tiny-smoke: Instant 20-step CPU/GPU verification on synthetic data (~1-2 seconds)
- --preset: Architectural preset (bravien-tiny, bravien-small, bravien-1.0b, bravien-1.5b)
- --data-path: Path to packed pretraining dataset file (.pt)
- Full distributed & local configuration, loss logging, and performance reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from bravien.data.packer import PackedPretrainingDataset
from bravien.model.bravien_config import BRAVIEN_PRESETS, get_bravien_preset
from bravien.model.bravien_model import BravienForCausalLM
from bravien.training.pretrain import BravienPretrainer
from bravien.training.pretrain_config import PretrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a native Bravien Transformer model.")
    parser.add_argument(
        "--preset",
        type=str,
        default="bravien-tiny",
        choices=list(BRAVIEN_PRESETS.keys()),
        help="Architectural preset for the model.",
    )
    parser.add_argument(
        "--tiny-smoke",
        action="store_true",
        help="Run a fast 20-step smoke test on tiny architecture to verify training mechanics.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to packed dataset .pt file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Total pretraining steps.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Micro batch size per accumulation step.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of micro-batches to accumulate before optimizer step.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/bravien-native-1.5b",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "no"],
        help="Precision mode for training.",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="reports/bravien_native_1p5b_training.json",
        help="Path to save training summary JSON report.",
    )

    args = parser.parse_args()

    # Configure smoke test mode or target pretrain
    if args.tiny_smoke:
        preset_name = "bravien-tiny"
        max_steps = args.max_steps or 20
        log_interval = 5
        save_interval = 10
        out_dir = "checkpoints/native_smoke_test"
    else:
        preset_name = args.preset
        max_steps = args.max_steps or 100
        log_interval = 10
        save_interval = 50
        out_dir = args.output_dir

    # Load dataset if specified
    dataset = None
    if args.data_path and Path(args.data_path).exists():
        print(f"Loading packed dataset from {args.data_path}...")
        dataset = PackedPretrainingDataset.load(args.data_path)
        print(f"Loaded {len(dataset):,} packed blocks.")

    model_config = get_bravien_preset(preset_name)
    if dataset is not None:
        # Ensure max_position_embeddings and sequence length match dataset
        if dataset.max_seq_len > model_config.max_position_embeddings:
            model_config.max_position_embeddings = dataset.max_seq_len

    model = BravienForCausalLM(model_config)
    param_report = model.count_parameters()

    train_config = PretrainConfig(
        learning_rate=args.lr,
        max_steps=max_steps,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_interval=log_interval,
        save_interval=save_interval,
        output_dir=out_dir,
        mixed_precision=args.mixed_precision,
        tiny_smoke_mode=args.tiny_smoke,
    )

    pretrainer = BravienPretrainer(model=model, config=train_config, train_dataset=dataset)
    results = pretrainer.train()

    # Save comprehensive training report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_config.name,
        "parameter_count": param_report.total,
        "trainable_parameters": param_report.trainable,
        "dataset_source": args.data_path or "synthetic_curated_seed",
        "tokens_trained": results["total_tokens"],
        "training_duration_seconds": round(results["elapsed_seconds"], 2),
        "final_loss": round(results["final_loss"], 4),
        "device": str(pretrainer.device),
        "mixed_precision": args.mixed_precision,
        "checkpoint_directory": out_dir,
        "status": "COMPLETED",
    }

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nSaved training report to: {report_path}")
    print(f"Training session successfully completed!\n")


if __name__ == "__main__":
    main()
