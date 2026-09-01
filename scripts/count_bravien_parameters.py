"""Bravien Model Parameter Calculator and Configuration Explorer.

Computes exact parameter counts across embeddings, attention, MLP, normalization,
and output head for any arbitrary or preset Bravien Transformer configuration.
Verifies programmatic count against tensor model instantiation and memory math.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from bravien.model.bravien_config import BRAVIEN_PRESETS, BravienConfig
from bravien.model.bravien_model import BravienForCausalLM


def calculate_analytical_params(config: BravienConfig) -> dict[str, int]:
    """Calculate exact theoretical parameter counts mathematically."""
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    intermediate_size = config.intermediate_size

    # Embeddings: vocab_size * hidden_size
    embedding_params = vocab_size * hidden_size

    # Per-layer attention:
    # Q: hidden_size * (num_heads * head_dim)
    # K: hidden_size * (num_kv_heads * head_dim)
    # V: hidden_size * (num_kv_heads * head_dim)
    # O: (num_heads * head_dim) * hidden_size
    q_params = hidden_size * (num_heads * head_dim)
    k_params = hidden_size * (num_kv_heads * head_dim)
    v_params = hidden_size * (num_kv_heads * head_dim)
    o_params = (num_heads * head_dim) * hidden_size
    layer_attn_params = q_params + k_params + v_params + o_params
    total_attn_params = layer_attn_params * num_layers

    # Per-layer MLP (SwiGLU):
    # gate_proj: hidden_size * intermediate_size
    # up_proj: hidden_size * intermediate_size
    # down_proj: intermediate_size * hidden_size
    layer_mlp_params = 3 * (hidden_size * intermediate_size)
    total_mlp_params = layer_mlp_params * num_layers

    # Per-layer Norms:
    # input_norm: hidden_size
    # post_attn_norm: hidden_size
    # final_norm: hidden_size
    norm_params = (2 * hidden_size * num_layers) + hidden_size

    # Output Head:
    lm_head_params = 0 if config.tie_word_embeddings else (hidden_size * vocab_size)

    total = embedding_params + total_attn_params + total_mlp_params + norm_params + lm_head_params

    return {
        "embedding": embedding_params,
        "attention": total_attn_params,
        "mlp": total_mlp_params,
        "norm": norm_params,
        "lm_head": lm_head_params,
        "total": total,
    }


def print_comparison_table() -> None:
    """Print an architectural comparison of all Bravien candidate presets."""
    print("\n" + "=" * 105)
    print("BRAVIEN ARCHITECTURAL CONFIGURATION EXPLORATION (~1.0B to ~1.7B TARGETS)")
    print("=" * 105)

    headers = f"{'Preset':<14} | {'Layers':<6} | {'Hidden':<6} | {'Heads (Q/KV)':<12} | {'Interm.':<8} | {'Vocab':<6} | {'Tied?':<6} | {'Params (M)':<10} | {'Params (B)':<10}"
    print(headers)
    print("-" * 105)

    for name, config in BRAVIEN_PRESETS.items():
        counts = calculate_analytical_params(config)
        total_m = counts["total"] / 1e6
        total_b = counts["total"] / 1e9
        tied_str = "Yes" if config.tie_word_embeddings else "No"
        heads_str = f"{config.num_heads}/{config.num_kv_heads}"

        print(
            f"{name:<14} | {config.num_layers:<6} | {config.hidden_size:<6} | "
            f"{heads_str:<12} | {config.intermediate_size:<8} | {config.vocab_size:<6} | "
            f"{tied_str:<6} | {total_m:>9.2f}M | {total_b:>9.3f}B"
        )
    print("=" * 105 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate parameters for Bravien architectures.")
    parser.add_argument(
        "--preset",
        type=str,
        default="bravien-1.5b",
        choices=list(BRAVIEN_PRESETS.keys()),
        help="Preset architecture name to analyze.",
    )
    parser.add_argument("--compare", action="store_true", help="Print comparison of all presets.")
    parser.add_argument("--instantiate-check", action="store_true", help="Instantiate PyTorch model in memory and count tensors directly.")

    args = parser.parse_args()

    if args.compare:
        print_comparison_table()

    config = BRAVIEN_PRESETS[args.preset]
    counts = calculate_analytical_params(config)
    total = counts["total"]

    print(f"\nBravien architecture parameter report")
    print(f"-------------------------------------")
    print(f"Preset Name:       {config.name}")
    print(f"Layers:            {config.num_layers}")
    print(f"Hidden size:       {config.hidden_size}")
    print(f"Attention heads:   {config.num_heads} (Head Dim: {config.head_dim})")
    print(f"KV heads:          {config.num_kv_heads} (GQA: {config.num_heads // config.num_kv_heads}:1)")
    print(f"FFN size:          {config.intermediate_size} (SwiGLU)")
    print(f"Vocabulary:        {config.vocab_size:,}")
    print(f"Max Context:       {config.max_position_embeddings}")
    print(f"Tied Embeddings:   {config.tie_word_embeddings}")
    print(f"-------------------------------------")
    print(f"Embedding parameters:  {counts['embedding']:,} ({counts['embedding']/total*100:.1f}%)")
    print(f"Attention parameters:  {counts['attention']:,} ({counts['attention']/total*100:.1f}%)")
    print(f"FFN parameters:        {counts['mlp']:,} ({counts['mlp']/total*100:.1f}%)")
    print(f"Norm parameters:       {counts['norm']:,}")
    print(f"LM head parameters:    {counts['lm_head']:,} ({'Tied' if config.tie_word_embeddings else 'Untied'})")
    print(f"-------------------------------------")
    print(f"TOTAL PARAMETERS:      {total:,} ({total/1e9:.3f}B)")
    print(f"-------------------------------------")
    print(f"Estimated Weights Memory:")
    print(f"  - FP32 (4 bytes/param):   {total * 4 / (1024**3):.2f} GB")
    print(f"  - BF16/FP16 (2 B/param):  {total * 2 / (1024**3):.2f} GB")
    print(f"  - INT8 (1 byte/param):    {total * 1 / (1024**3):.2f} GB")
    print(f"  - INT4 (0.5 byte/param):  {total * 0.5 / (1024**3):.2f} GB")
    print(f"-------------------------------------")

    # Range target check (1.45B - 1.60B for 1.5B preset)
    if "1.5b" in args.preset.lower():
        min_target = 1_450_000_000
        max_target = 1_600_000_000
        is_pass = min_target <= total <= max_target
        status_str = "PASS" if is_pass else "FAIL"
        print(f"Target: ~1.5B (1.45B - 1.60B)")
        print(f"Status: {status_str}")
        if not is_pass:
            raise ValueError(f"Total parameter count {total:,} is outside the required target window [1.45B - 1.60B]!")

    if args.instantiate_check:
        print("\nInstantiating PyTorch model to verify exact parameter tensor count...")
        import torch
        with torch.device("meta"):
            meta_model = BravienForCausalLM(config)
            tensor_params = sum(p.numel() for p in meta_model.parameters() if p.requires_grad)
        print(f"PyTorch Verified Trainable Parameters: {tensor_params:,}")
        assert tensor_params == total, f"Discrepancy: Analytical {total} != PyTorch {tensor_params}"
        print("Verification: 100% Exact Match.")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
