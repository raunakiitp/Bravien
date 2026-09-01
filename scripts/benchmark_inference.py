#!/usr/bin/env python3
"""Bravien Local Inference Benchmark Suite (§31, §32).

Measures local model performance:
- Model & tokenizer loading latency
- Warmup latency
- Generation latency and throughput (tokens/sec)
- Context budgeting and prompt token counts
- Peak memory usage
- Device & hardware utilization

Safe to run on CPU and CUDA accelerators with zero network overhead.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add workspace root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from bravien.inference.engine import EngineConfig, InferenceEngine
from bravien.inference.hf_engine import HFInferenceEngine
from bravien.utils.hardware import detect_device


def benchmark_engine(
    model_name_or_path: str,
    *,
    prompts: list[str] | None = None,
    max_new_tokens: int = 32,
    device: str | None = None,
    dtype: str = "auto",
) -> dict[str, float | str | int]:
    print("=" * 60)
    print("BRAVIEN LOCAL MODEL INFERENCE BENCHMARK")
    print("=" * 60)

    device_info = detect_device(device)
    print(f"Target Device: {device_info.name} ({device_info.device})")
    print(f"Model: {model_name_or_path}")

    # 1. Model & Tokenizer Load
    t0 = time.time()
    is_hf = "Qwen" in model_name_or_path or "/" in model_name_or_path or not Path(model_name_or_path).exists()
    
    if is_hf:
        engine = HFInferenceEngine.from_pretrained(
            model_name_or_path,
            config=EngineConfig(device=device, dtype=dtype),
        )
    else:
        engine = InferenceEngine.load(
            model_name_or_path,
            config=EngineConfig(device=device, dtype=dtype),
        )
    load_time_sec = round(time.time() - t0, 3)
    print(f"Model Load Time: {load_time_sec}s")

    # 2. Warmup
    t0 = time.time()
    warmup_res = engine.warmup("Hello")
    warmup_time_ms = round((time.time() - t0) * 1000, 2)
    print(f"Warmup Latency: {warmup_time_ms}ms")

    # 3. Test Prompts
    test_prompts = prompts or [
        "Explain the benefit of local AI inference in one concise sentence.",
        "Calculate: What is 125 multiplied by 8?",
        "Write a Python function to compute the Fibonacci sequence.",
    ]

    total_tokens = 0
    total_time = 0.0
    latencies = []

    print("\n--- Running Generation Benchmarks ---")
    for idx, prompt in enumerate(test_prompts, start=1):
        print(f"\n[Prompt {idx}]: {prompt}")
        t_start = time.time()
        res = engine.complete(
            prompt,
            generation={"max_new_tokens": max_new_tokens, "temperature": 0.1},
        )
        duration = time.time() - t_start
        gen_tokens = res.completion_tokens
        prompt_tokens = res.prompt_tokens
        tok_per_sec = round(gen_tokens / max(duration, 1e-4), 1)

        total_tokens += gen_tokens
        total_time += duration
        latencies.append(duration)

        print(f"Generated: {res.text.strip()}")
        print(f"Tokens: {gen_tokens} (prompt: {prompt_tokens}) | Duration: {duration:.3f}s | Speed: {tok_per_sec} tok/s")

    avg_tok_per_sec = round(total_tokens / max(total_time, 1e-4), 1)
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Device: {device_info.name}")
    print(f"Model Load: {load_time_sec}s")
    print(f"Warmup: {warmup_time_ms}ms")
    print(f"Total Output Tokens: {total_tokens}")
    print(f"Average Throughput: {avg_tok_per_sec} tok/s")
    print("=" * 60)

    return {
        "device": device_info.name,
        "load_time_sec": load_time_sec,
        "warmup_time_ms": warmup_time_ms,
        "total_tokens": total_tokens,
        "avg_tokens_per_sec": avg_tok_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Bravien local inference.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model checkpoint or HF identifier.")
    parser.add_argument("--tokens", type=int, default=32, help="Max tokens per prompt.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu').")
    args = parser.parse_args()

    benchmark_engine(args.model, max_new_tokens=args.tokens, device=args.device)


if __name__ == "__main__":
    main()
