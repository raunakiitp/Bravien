"""CLI tool to inspect Bravien model architecture, metadata, and parameter counts."""

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

from bravien.model.bravien_config import BRAVIEN_PRESETS, BravienConfig, get_bravien_preset
from bravien.model.bravien_model import BravienForCausalLM
from bravien.model.parameter_count import count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Bravien Model Architecture and Parameters.")
    parser.add_argument(
        "--model",
        "--preset",
        type=str,
        default="bravien-1.5b",
        help="Model preset name (e.g. bravien-1.5b, bravien-small, bravien-tiny) or path to checkpoint.",
    )
    args = parser.parse_args()

    model_arg = args.model
    if model_arg in BRAVIEN_PRESETS:
        config = get_bravien_preset(model_arg)
    elif Path(model_arg).exists() and (Path(model_arg) / "config.json").exists():
        config = BravienConfig.from_json_file(Path(model_arg) / "config.json")
    else:
        config = get_bravien_preset("bravien-1.5b")

    # Instantiate on meta device to inspect zero-RAM structure
    with torch.device("meta"):
        model = BravienForCausalLM(config)
        report = count_parameters(model)

    print("=" * 65)
    print(f"BRAVIEN ARCHITECTURE INSPECTION REPORT")
    print("=" * 65)
    print(f"Model:                {config.name.capitalize()}")
    print(f"Architecture:         BravienForCausalLM")
    print(f"Model Type:           {config.model_type}")
    print(f"Parameters:           ~{report['total_billions']}B ({report['total_parameters']:,})")
    print(f"Trainable Parameters: {report['trainable_parameters']:,}")
    print(f"Vocabulary Size:      {config.vocab_size:,}")
    print(f"Layers:               {config.num_layers}")
    print(f"Hidden Size:          {config.hidden_size}")
    print(f"Attention Heads:      {config.num_heads} (KV Heads: {config.num_kv_heads}, GQA: {config.num_heads // config.num_kv_heads}:1)")
    print(f"Intermediate Size:    {config.intermediate_size} (SwiGLU)")
    print(f"Context Length:       {config.max_position_embeddings}")
    print(f"Positional Encoding:  RoPE (theta={config.rope_theta})")
    print(f"Normalization:        {config.norm_kind.upper()} (eps={config.norm_eps})")
    print(f"Word Embeddings Tied: {config.tie_word_embeddings}")
    print("-" * 65)
    print("Parameter Distribution:")
    print(f"  - Embeddings:       {report['embedding_parameters']:,} ({report['embedding_parameters']/report['total_parameters']*100:.1f}%)")
    print(f"  - Self-Attention:   {report['attention_parameters']:,} ({report['attention_parameters']/report['total_parameters']*100:.1f}%)")
    print(f"  - SwiGLU MLP:       {report['mlp_parameters']:,} ({report['mlp_parameters']/report['total_parameters']*100:.1f}%)")
    print(f"  - RMSNorm Layers:   {report['norm_parameters']:,}")
    print(f"  - Output LM Head:   {report['lm_head_parameters']:,} ({'Tied' if config.tie_word_embeddings else 'Untied'})")
    print("-" * 65)
    print("Weights Memory Footprint:")
    print(f"  - BF16 / FP16:      {report['memory_weights_gb']['bf16']} GB")
    print(f"  - INT8 Quantized:   {report['memory_weights_gb']['int8']} GB")
    print(f"  - INT4 Quantized:   {report['memory_weights_gb']['int4']} GB")
    print(f"Target Status:        {report['status']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
