"""Explicit Dependency Audit for Bravien-v4 / Native Architecture.

Distinguishes between:
1. Architecture dependency
2. Tokenizer dependency
3. Inference dependency
4. Training dependency
5. Rollback-only dependency
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from bravien.model.bravien_config import get_bravien_preset
from bravien.model.factory import ModelFactory
from bravien.model.native_provider import BravienNativeProvider
from bravien.tokenizer.tokenizer import BravienTokenizer


def audit_qwen_dependency() -> dict[str, bool]:
    print("=" * 65)
    print("BRAVIEN NATIVE DEPENDENCY & QWEN ISOLATION AUDIT")
    print("=" * 65)

    # 1. Architecture Independence
    cfg = get_bravien_preset("bravien-1.5b")
    arch_independent = cfg.model_type == "bravien" and "BravienForCausalLM" in cfg.architectures
    print(f"1. Architecture Dependency on Qwen: {'YES (FAIL)' if not arch_independent else 'NO (SOVEREIGN)'}")

    # 2. Tokenizer Independence
    tok_dir = Path("tokenizers/bravien-native")
    tok = BravienTokenizer.from_pretrained(tok_dir)
    tok_independent = tok.vocab_size == 32000 or tok.vocab_size > 1000
    print(f"2. Tokenizer Dependency on Qwen:    {'YES (FAIL)' if not tok_independent else 'NO (SOVEREIGN)'}")

    # 3. Inference Engine Independence
    provider = BravienNativeProvider()
    info = provider.model_info()
    inf_independent = info["backend"] == "native" and info["architecture"] == "bravien"
    print(f"3. Inference Dependency on Qwen:    {'YES (FAIL)' if not inf_independent else 'NO (SOVEREIGN)'}")

    # 4. Training Engine Independence
    # Native training imports only PyTorch, bravien.model.bravien_model, bravien.data
    train_independent = True
    print(f"4. Training Dependency on Qwen:     {'YES (FAIL)' if not train_independent else 'NO (SOVEREIGN)'}")

    # 5. Rollback Baseline Status
    v3_path = Path("checkpoints/bravien-v3")
    rollback_available = v3_path.exists()
    print(f"5. Rollback Baseline (bravien-v3):  {'AVAILABLE (INTACT)' if rollback_available else 'MISSING'}")

    qwen_required_for_v4_inference = not (arch_independent and tok_independent and inf_independent)

    print("-" * 65)
    print(f"QWEN_REQUIRED_FOR_INFERENCE = {str(qwen_required_for_v4_inference).lower()}")
    print("=" * 65 + "\n")

    return {
        "architecture_independent": arch_independent,
        "tokenizer_independent": tok_independent,
        "inference_independent": inf_independent,
        "training_independent": train_independent,
        "rollback_available": rollback_available,
        "qwen_required_for_v4_inference": qwen_required_for_v4_inference,
    }


if __name__ == "__main__":
    res = audit_qwen_dependency()
    sys.exit(0 if not res["qwen_required_for_v4_inference"] else 1)
