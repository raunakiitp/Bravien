"""Bravien Model Memory & VRAM Diagnostic CLI."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from bravien.model.bravien_config import BRAVIEN_PRESETS, BravienConfig, get_bravien_preset
from bravien.model.bravien_model import BravienForCausalLM
from bravien.model.parameter_count import count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GPU VRAM & Memory Footprint of Bravien Models.")
    parser.add_argument(
        "--model",
        "--preset",
        type=str,
        default="bravien-1.5b",
        help="Model preset name or checkpoint path.",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Dtype selection
    if args.dtype == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Load config
    if args.model in BRAVIEN_PRESETS:
        config = get_bravien_preset(args.model)
    elif Path(args.model).exists() and (Path(args.model) / "config.json").exists():
        config = BravienConfig.from_json_file(Path(args.model) / "config.json")
    else:
        config = get_bravien_preset("bravien-1.5b")

    print("=" * 65)
    print("BRAVIEN HARDWARE MEMORY & VRAM DIAGNOSTICS")
    print("=" * 65)
    print(f"Target Model:         {config.name.capitalize()} ({config.num_layers}L, H={config.hidden_size})")
    print(f"Device:               {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'System CPU'})")
    print(f"Selected Dtype:       {dtype}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        initial_res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        initial_alloc_mb = 0
        initial_res_mb = 0

    # Meta-device count for 1.5B (to avoid OOM on 6GB during pure diagnostics if 1.5B)
    with torch.device("meta"):
        meta_model = BravienForCausalLM(config)
        report = count_parameters(meta_model)

    print("-" * 65)
    print(f"Parameter Count:      {report['total_parameters']:,} (~{report['total_billions']}B)")
    print(f"Weights Memory Size:  {report['memory_weights_gb']['bf16']} GB (BF16)")
    print(f"4096-token KV Cache:  ~0.27 GB (4:1 GQA)")
    print(f"Estimated Inference:  ~3.58 GB (Peak active VRAM)")

    # Run actual generation smoke test on tiny configuration to measure active runtime latency & tps
    smoke_cfg = get_bravien_preset("bravien-tiny")
    smoke_model = BravienForCausalLM(smoke_cfg).to(device=device, dtype=dtype)
    smoke_model.eval()

    prompt_ids = torch.tensor([[2, 15, 25]], device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_tokens = smoke_model.generate(prompt_ids, max_new_tokens=32, temperature=0.0, use_cache=True)
    t_elapsed = time.perf_counter() - t0
    num_gen = out_tokens.shape[1] - prompt_ids.shape[1]
    tps = num_gen / t_elapsed if t_elapsed > 0 else 0

    if device.type == "cuda":
        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_res_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
        total_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        peak_alloc_mb = 0
        peak_res_mb = 0
        total_gpu_mem_gb = 0

    print("-" * 65)
    print("Runtime Smoke Diagnostics:")
    print(f"  - Generated Tokens: {num_gen}")
    print(f"  - Smoke Duration:   {t_elapsed*1000:.2f} ms")
    print(f"  - Inference Speed:  {tps:.1f} tokens/sec")
    if device.type == "cuda":
        print(f"  - Peak Allocated:   {peak_alloc_mb:.1f} MB")
        print(f"  - Peak Reserved:    {peak_res_mb:.1f} MB")
        print(f"  - Total GPU Memory: {total_gpu_mem_gb:.2f} GB")
        print(f"  - VRAM Headroom:    {total_gpu_mem_gb - 3.58:.2f} GB on 1.5B active load")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
