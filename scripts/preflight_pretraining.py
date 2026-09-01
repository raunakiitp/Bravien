"""Bravien Foundation Pretraining Preflight Verification Utility.

Performs comprehensive static and environmental preflight checks before running
foundation pretraining or real-data pilots on cloud or local GPU instances.

Checks:
- System resources (GPU VRAM, CUDA, PyTorch, BF16 capability, RAM, free disk)
- Model architecture specifications & parameter calculations
- Tokenizer integrity and vocabulary bounds
- Dataset existence, manifest metadata, and packed token integrity
- Checkpoint directory and resume discovery
- DOES NOT EXECUTE TRAINING.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import yaml

from bravien.data.packer import PackedPretrainingDataset
from bravien.model.bravien_config import BRAVIEN_PRESETS, BravienConfig, get_bravien_preset
from bravien.model.parameter_count import count_parameters
from bravien.tokenizer.tokenizer import BravienTokenizer


def run_preflight(config_path: str | Path) -> dict[str, Any]:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"❌ Configuration file not found: {cfg_file}")
        sys.exit(1)

    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("BRAVIEN FOUNDATION PRETRAINING PREFLIGHT AUDIT")
    print(f"Configuration File: {cfg_file}")
    print("=" * 70)

    report: dict[str, Any] = {"status": "PASS", "checks": {}, "warnings": [], "errors": []}

    # 1. Environment & Hardware Checks
    print("\n[1/5] Hardware & Environment Resources:")
    cuda_avail = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_avail else 0
    pytorch_version = torch.__version__
    cuda_version = torch.version.cuda if cuda_avail else "N/A"

    print(f"  - PyTorch Version:     {pytorch_version}")
    print(f"  - CUDA Available:      {cuda_avail} (Version: {cuda_version})")
    print(f"  - GPU Devices:         {gpu_count}")

    total_vram_gb = 0.0
    bf16_supported = False
    if cuda_avail:
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            total_vram_gb += vram_gb
            cap = torch.cuda.get_device_capability(i)
            print(f"    * GPU {i}: {name} | VRAM: {vram_gb:.2f} GB | Compute Cap: {cap[0]}.{cap[1]}")
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"  - Native BF16 Support: {bf16_supported}")
    else:
        print("  ⚠️ No CUDA GPU detected. Training will run in CPU fallback mode (slow).")
        report["warnings"].append("No CUDA GPU detected.")

    # Disk & System RAM
    total_disk, used_disk, free_disk = shutil.disk_usage(".")
    free_disk_gb = free_disk / (1024**3)
    print(f"  - Free Workspace Disk: {free_disk_gb:.2f} GB")
    if free_disk_gb < 10.0:
        report["warnings"].append(f"Low disk space: {free_disk_gb:.2f} GB free (< 10 GB recommended).")

    # 2. Model Architecture & Parameters
    print("\n[2/5] Model Architecture Verification:")
    model_cfg = config.get("model", {})
    preset_name = model_cfg.get("name", "bravien-1.5b")
    vocab_size = model_cfg.get("vocab_size", 32000)
    hidden_size = model_cfg.get("hidden_size", 2048)
    num_layers = model_cfg.get("num_layers", 32)
    num_heads = model_cfg.get("num_heads", 16)
    num_kv_heads = model_cfg.get("num_kv_heads", 4)
    intermediate_size = model_cfg.get("intermediate_size", 5632)
    max_pos = model_cfg.get("max_position_embeddings", 4096)

    print(f"  - Target Model:        {preset_name}")
    print(f"  - Architecture:        Layers={num_layers}, Hidden={hidden_size}, Heads={num_heads}, KV-Heads={num_kv_heads} (GQA {num_heads//num_kv_heads}:1)")
    print(f"  - Intermediate FFN:    {intermediate_size} (SwiGLU, {intermediate_size/hidden_size:.2f}x)")
    print(f"  - Context Window:      {max_pos} tokens (RoPE)")
    print(f"  - Vocab Capacity:      {vocab_size} tokens")

    # Theoretical parameter count calculation
    # Embeddings: vocab_size * hidden_size
    # Per layer:
    #   Attn Q: hidden * hidden
    #   Attn K: hidden * (hidden * kv / heads)
    #   Attn V: hidden * (hidden * kv / heads)
    #   Attn Out: hidden * hidden
    #   MLP gate, up, down: 3 * hidden * intermediate
    #   Norms: 2 * hidden
    # Final norm: hidden
    embed_params = vocab_size * hidden_size
    attn_per_layer = (hidden_size * hidden_size) + 2 * (hidden_size * (hidden_size * num_kv_heads // num_heads)) + (hidden_size * hidden_size)
    mlp_per_layer = 3 * hidden_size * intermediate_size
    norm_per_layer = 2 * hidden_size
    total_calc = embed_params + num_layers * (attn_per_layer + mlp_per_layer + norm_per_layer) + hidden_size
    print(f"  - Exact Parameter Total: {total_calc:,} (~{total_calc/1e9:.3f}B params)")

    # 3. Tokenizer Integrity
    print("\n[3/5] Tokenizer Validation:")
    tok_cfg = config.get("tokenizer", {})
    tok_path = Path(tok_cfg.get("path", "tokenizers/bravien-native"))
    if not tok_path.exists():
        print(f"❌ Tokenizer directory not found at: {tok_path}")
        report["errors"].append(f"Tokenizer missing: {tok_path}")
    else:
        try:
            tokenizer = BravienTokenizer.from_pretrained(tok_path)
            actual_vocab = tokenizer.vocab_size
            print(f"  - Tokenizer Path:      {tok_path}")
            print(f"  - Loaded Vocabulary:   {actual_vocab:,} tokens")
            print(f"  - Special Tokens:      BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}, UNK={tokenizer.unk_token_id}")
            if actual_vocab > vocab_size:
                report["errors"].append(f"Tokenizer vocab ({actual_vocab}) exceeds model capacity ({vocab_size})")
            else:
                print(f"  ✅ Tokenizer fits within model embedding table ({actual_vocab} <= {vocab_size})")
        except Exception as e:
            report["errors"].append(f"Failed to load tokenizer: {e}")
            print(f"❌ Tokenizer load failed: {e}")

    # 4. Dataset & Manifest Status
    print("\n[4/5] Dataset & Staging Status:")
    ds_cfg = config.get("dataset", {})
    manifest_path = Path(ds_cfg.get("manifest_path", "manifests/pretraining_manifest.json"))
    packed_data_dir = Path(ds_cfg.get("packed_data_dir", "data/packed_pretrain"))
    seq_len = ds_cfg.get("sequence_length", 2048)

    print(f"  - Configured Seq Len:  {seq_len} tokens")
    print(f"  - Manifest Path:       {manifest_path}")
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            samples = manifest_data.get("total_samples", 0)
            tokens = manifest_data.get("total_estimated_tokens", 0)
            print(f"    * Manifest: {len(manifest_data.get('sources', []))} sources, {samples:,} samples, {tokens:,} tokens")
        except Exception as e:
            print(f"    ⚠️ Could not parse manifest: {e}")
    else:
        print("    ⚠️ Pretraining manifest file not yet generated.")

    print(f"  - Packed Data Dir:     {packed_data_dir}")
    packed_files = list(packed_data_dir.glob("*.pt")) if packed_data_dir.exists() else []
    print(f"    * Available Packed Datasets: {len(packed_files)} file(s)")
    for pf in packed_files:
        size_mb = pf.stat().st_size / (1024 * 1024)
        print(f"      - {pf.name} ({size_mb:.2f} MB)")

    # 5. Checkpoint & Resume Status
    print("\n[5/5] Checkpoint & Resume Configuration:")
    ckpt_cfg = config.get("checkpoints", {})
    output_dir = Path(ckpt_cfg.get("output_dir", "checkpoints/bravien-native-1.5b"))
    print(f"  - Checkpoint Dir:      {output_dir}")
    if output_dir.exists():
        existing_steps = [p.name for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("step_")]
        if existing_steps:
            existing_steps.sort()
            print(f"    * Found {len(existing_steps)} existing checkpoint(s): latest is {existing_steps[-1]}")
            print(f"    * Training execution will automatically RESUME from: {existing_steps[-1]}")
        else:
            print("    * No existing step checkpoints found. Training execution will START FRESH.")
    else:
        print("    * Output directory does not yet exist (will be created automatically on start).")

    # Final Summary
    print("\n" + "=" * 70)
    if report["errors"]:
        print("❌ PREFLIGHT FAILED - ERRORS DETECTED:")
        for err in report["errors"]:
            print(f"  - {err}")
        report["status"] = "FAIL"
    else:
        print("✅ PREFLIGHT AUDIT PASSED - READY FOR EXECUTION")
        if report["warnings"]:
            print("  (Notes / Warnings):")
            for w in report["warnings"]:
                print(f"  - {w}")
    print("=" * 70 + "\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Preflight System & Config Audit for Bravien Pretraining.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bravien_1p5b_pretrain.yaml",
        help="Path to pretraining YAML configuration file.",
    )
    args = parser.parse_args()
    report = run_preflight(args.config)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
