"""Stage 10 Architecture Verification Runner CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from bravien.model.bravien_config import get_bravien_preset
from bravien.model.bravien_model import BravienForCausalLM
from bravien.model.factory import ModelFactory
from bravien.model.parameter_count import count_parameters
from bravien.tokenizer.tokenizer import BravienTokenizer


def run_stage10_tests() -> bool:
    print("=" * 70)
    print("BRAVIEN STAGE 10: NATIVE 1.5B ARCHITECTURE VERIFICATION")
    print("=" * 70)

    cfg = get_bravien_preset("bravien-1.5b")

    # 1. Parameter count check
    print("\n1. Parameter Count Verification:")
    with torch.device("meta"):
        model = BravienForCausalLM(cfg)
        rep = count_parameters(model)
    print(f"   [PASS] Total Parameters: {rep['total_parameters']:,} (~{rep['total_billions']}B)")
    print(f"   [PASS] Target Status:    {rep['status']}")
    assert rep["is_target_1p5b"], f"Parameters out of target window: {rep['total_parameters']}"

    # 2. Independence from Qwen
    print("\n2. Model Sovereignty & Identity:")
    print(f"   [PASS] model_type:       {cfg.model_type}")
    print(f"   [PASS] architecture:     {cfg.architecture}")
    assert cfg.model_type == "bravien"
    assert cfg.model_type != "qwen2"
    assert cfg.architecture == "BravienForCausalLM"

    # 3. ModelFactory resolution
    print("\n3. Dynamic Model Factory Resolution:")
    tiny_m = ModelFactory.create_model("bravien-tiny", device="cpu")
    print(f"   [PASS] Successfully instantiated {type(tiny_m).__name__} via ModelFactory.")

    # 4. Tokenizer verification
    print("\n4. Tokenizer Asset Verification:")
    tok = BravienTokenizer.from_pretrained(Path("tokenizers/bravien-native"))
    print(f"   [PASS] Loaded BravienTokenizer with {tok.vocab_size:,} vocabulary.")

    print("\n" + "=" * 70)
    print("STAGE 10 ARCHITECTURAL VERIFICATION: ALL GATES PASSED")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    success = run_stage10_tests()
    sys.exit(0 if success else 1)
