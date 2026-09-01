"""CLI tool for executing alignment and preference reinforcement on Bravien checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch

from bravien.model.checkpoint import load_bravien_checkpoint, save_bravien_checkpoint
from bravien.tokenizer.tokenizer import BravienTokenizer
from bravien.training.alignment import BravienAligner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preference alignment on a Bravien checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/native_smoke_test/step_0000030", help="Path to checkpoint.")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizers/bravien-native")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="checkpoints/bravien-v4-alignment")

    args = parser.parse_args()

    print("=" * 65)
    print("BRAVIEN CAPABILITY REINFORCEMENT & PREFERENCE ALIGNMENT")
    print("=" * 65)
    print(f"Input Checkpoint:    {args.checkpoint}")
    print(f"Tokenizer Directory: {args.tokenizer_dir}")
    print(f"Target Epochs:       {args.epochs}")
    print(f"Output Directory:    {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device:    {device}")

    model = load_bravien_checkpoint(Path(args.checkpoint), device=device)
    tokenizer = BravienTokenizer.from_pretrained(Path(args.tokenizer_dir))

    aligner = BravienAligner(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    results = aligner.train_alignment(num_epochs=args.epochs)
    print("\n[PASS] Alignment Optimization Complete:")
    print(f"  - Preference Pairs: {results['total_pairs']}")
    print(f"  - Average Loss:     {results['average_loss']:.4f}")
    print(f"  - Final Loss:       {results['final_loss']:.4f}")

    out_path = Path(args.output_dir)
    save_bravien_checkpoint(model, out_path)
    print(f"  - Saved Aligned Model to: {out_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
