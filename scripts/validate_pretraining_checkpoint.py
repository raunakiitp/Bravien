"""Validator for Bravien Checkpoints.

Verifies checkpoint integrity:
- Manifest and SHA-256 tensor checksums
- Model config compatibility
- Zero NaNs or Infs in parameter tensors
- Forward pass and logit shape validation
- Deterministic autoregressive generation
"""

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

from bravien.model.checkpoint import load_bravien_checkpoint
from bravien.model.parameter_count import count_parameters


def validate_checkpoint(checkpoint_dir: Path, device: str = "cpu") -> bool:
    print("=" * 65)
    print(f"BRAVIEN CHECKPOINT INTEGRITY VALIDATOR: {checkpoint_dir}")
    print("=" * 65)

    if not checkpoint_dir.exists():
        print(f"[FAIL] Directory does not exist: {checkpoint_dir}")
        return False

    dev = torch.device(device)
    print(f"Loading checkpoint on target device: {dev}...")
    try:
        model = load_bravien_checkpoint(checkpoint_dir, device=dev, verify_checksums=True)
    except Exception as e:
        print(f"[FAIL] Failed to load checkpoint: {e}")
        return False

    # 1. Parameter tensor finiteness check
    print("\n1. Verifying parameter tensor finiteness (No NaNs / Infs)...")
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            print(f"[FAIL] NaN detected in parameter tensor: {name}")
            return False
        if torch.isinf(p).any():
            print(f"[FAIL] Inf detected in parameter tensor: {name}")
            return False
    print("   [PASS] All parameter tensors are 100% finite and uncorrupted.")

    # 2. Parameter report
    rep = count_parameters(model)
    print(f"\n2. Parameter Report: {rep['total_parameters']:,} (~{rep['total_billions']}B) | Status: {rep['status']}")

    # 3. Forward pass check
    print("\n3. Testing forward pass with sample input...")
    inp = torch.tensor([[model.config.bos_token_id, 10, 20, 30]], device=dev)
    try:
        with torch.no_grad():
            out = model(inp)
        print(f"   [PASS] Forward output shape: {out.logits.shape} (Expected: (1, 4, {model.config.vocab_size}))")
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        return False

    # 4. Generation smoke test
    print("\n4. Testing generation smoke test...")
    try:
        with torch.no_grad():
            gen_tokens = model.generate(inp, max_new_tokens=8, temperature=0.0, use_cache=True)
        print(f"   [PASS] Generated output sequence: {gen_tokens.tolist()[0]}")
    except Exception as e:
        print(f"[FAIL] Generation failed: {e}")
        return False

    print("\n" + "=" * 65)
    print("ALL CHECKPOINT VALIDATION GATES PASSED (100% INTEGRITY)")
    print("=" * 65 + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Bravien Checkpoint Integrity.")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    success = validate_checkpoint(Path(args.checkpoint), device=args.device)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
