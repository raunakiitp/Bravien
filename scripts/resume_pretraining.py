"""CLI Tool for Resuming Pretraining from a Saved Checkpoint."""

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

from bravien.data.packer import PackedPretrainingDataset
from bravien.model.checkpoint import load_bravien_checkpoint
from bravien.training.pretrain import BravienPretrainer
from bravien.training.pretrain_config import PretrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume pretraining from an existing checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory to resume from.")
    parser.add_argument("--additional-steps", type=int, default=50, help="Number of additional steps to train.")
    parser.add_argument("--data-path", type=str, default="data/packed_pretrain/train_packed.pt")
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    print("=" * 65)
    print(f"RESUMING BRAVIEN PRETRAINING FROM: {ckpt_path}")
    print("=" * 65)

    model = load_bravien_checkpoint(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu")
    dataset = PackedPretrainingDataset.load(args.data_path) if Path(args.data_path).exists() else None

    cfg = PretrainConfig(
        max_steps=args.additional_steps,
        learning_rate=args.lr,
        micro_batch_size=1,
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        output_dir=str(ckpt_path.parent),
    )

    trainer = BravienPretrainer(
        model=model,
        config=cfg,
        train_dataset=dataset,
    )
    result = trainer.train()
    print(f"\n[PASS] Resumed training complete! Steps: {result.total_steps}, Loss: {result.final_loss:.4f}")


if __name__ == "__main__":
    import torch
    main()
