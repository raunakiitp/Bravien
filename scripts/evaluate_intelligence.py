"""Deterministic Intelligence Benchmark Evaluator for Bravien Models.

Evaluates any model checkpoint against the 12-category Stage 6 benchmark.

Usage:
    python scripts/evaluate_intelligence.py --model checkpoints/bravien-v1
    python scripts/evaluate_intelligence.py --model checkpoints/bravien-v2 --output reports/eval_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from bravien.data.curation import BRAVIEN_SYSTEM_PROMPT
from bravien.evaluation.benchmarks.stage6_benchmark import (
    BRAVIEN_BENCHMARK_VERSION as STAGE6_VERSION,
    get_stage6_benchmark_items,
)
from bravien.evaluation.benchmarks.stage7_benchmark import (
    BRAVIEN_STAGE7_BENCHMARK_VERSION,
    BenchmarkItem,
    get_stage7_benchmark_items,
)
from bravien.inference.hf_engine import HFInferenceEngine
from bravien.utils.logging import configure_stdout, get_logger, setup_logging

logger = get_logger("scripts.evaluate_intelligence")


def grade_response(item: BenchmarkItem, response_text: str) -> tuple[bool, str]:
    """Grade a model response against benchmark assertions."""
    text_lower = response_text.lower()

    # 1. Custom validator function if defined
    if item.validator is not None:
        passed, reason = item.validator(response_text)
        if not passed:
            return False, reason

    # 2. Forbidden substrings (negative constraints)
    for forbidden in item.forbidden:
        if forbidden.lower() in text_lower:
            return False, f"Contains forbidden text: '{forbidden}'"

    # 3. Expected substrings (positive assertions)
    if item.expected:
        found_any = any(exp.lower() in text_lower for exp in item.expected)
        if not found_any:
            return False, f"Missing expected concepts: {item.expected}"

    return True, "Passed assertions"


def run_benchmark(
    model_path: str,
    benchmark_name: str = "stage7",
    device: str | None = None,
    precision: str = "auto",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> dict[str, Any]:
    version = BRAVIEN_STAGE7_BENCHMARK_VERSION if benchmark_name == "stage7" else STAGE6_VERSION
    print(f"\n==================================================")
    print(f"BRAVIEN INTELLIGENCE BENCHMARK ({benchmark_name.upper()} v{version})")
    print(f"Target Model: {model_path}")
    print(f"==================================================\n")

    engine = HFInferenceEngine.from_pretrained(
        model_path,
        device=device,
        dtype="bf16" if precision in ("bfloat16", "bf16") else "auto",
    )

    items = get_stage7_benchmark_items() if benchmark_name == "stage7" else get_stage6_benchmark_items()
    results: list[dict[str, Any]] = []
    category_scores: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})

    t0_all = time.perf_counter()

    for i, item in enumerate(items, 1):
        messages = [
            {"role": "system", "content": BRAVIEN_SYSTEM_PROMPT},
            *item.history,
            {"role": "user", "content": item.prompt},
        ]

        t0 = time.perf_counter()
        gen_result = engine.chat(
            messages=messages,
            generation={"max_new_tokens": max_new_tokens, "temperature": temperature},
        )
        elapsed = time.perf_counter() - t0
        output_text = gen_result.text.strip()

        passed, reason = grade_response(item, output_text)
        category_scores[item.category]["total"] += 1
        if passed:
            category_scores[item.category]["passed"] += 1

        status_mark = "✅ PASS" if passed else "❌ FAIL"
        print(f"[{i:02d}/{len(items):02d}] {status_mark} | {item.category:<24} | {item.description}")
        if not passed:
            print(f"     Reason: {reason}")
            print(f"     Output: {repr(output_text[:120])}...")

        results.append({
            "id": item.id,
            "category": item.category,
            "description": item.description,
            "prompt": item.prompt,
            "output": output_text,
            "passed": passed,
            "reason": reason,
            "latency_ms": round(elapsed * 1000, 1),
            "completion_tokens": gen_result.completion_tokens,
            "prompt_tokens": gen_result.prompt_tokens,
        })

    total_elapsed = time.perf_counter() - t0_all
    total_passed = sum(c["passed"] for c in category_scores.values())
    total_items = len(items)
    overall_accuracy = (total_passed / total_items) * 100 if total_items > 0 else 0.0

    print("\n--------------------------------------------------")
    print("CATEGORY BREAKDOWN")
    print("--------------------------------------------------")
    category_summary = {}
    for cat, counts in category_scores.items():
        acc = (counts["passed"] / counts["total"]) * 100
        category_summary[cat] = {
            "passed": counts["passed"],
            "total": counts["total"],
            "accuracy": round(acc, 1),
        }
        print(f"{cat:<28}: {counts['passed']}/{counts['total']} ({acc:.1f}%)")

    print("\n==================================================")
    print(f"FINAL SCORE: {total_passed}/{total_items} ({overall_accuracy:.1f}%) in {total_elapsed:.2f}s")
    print("==================================================\n")

    report = {
        "benchmark": benchmark_name,
        "benchmark_version": version,
        "model": model_path,
        "overall_accuracy": round(overall_accuracy, 1),
        "total_passed": total_passed,
        "total_items": total_items,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "category_summary": category_summary,
        "results": results,
    }

    return report


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Evaluate model on Bravien Intelligence Benchmark.")
    parser.add_argument("--model", default="checkpoints/bravien-v2", help="Path to model checkpoint")
    parser.add_argument("--benchmark", default="stage7", choices=["stage6", "stage7"], help="Benchmark suite name")
    parser.add_argument("--device", default=None, help="Device to run on (cuda, cpu)")
    parser.add_argument("--precision", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--output", default=None, help="Output JSON path for report")
    args = parser.parse_args()

    report = run_benchmark(
        model_path=args.model,
        benchmark_name=args.benchmark,
        device=args.device,
        precision=args.precision,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved evaluation report to: {out_path}")


if __name__ == "__main__":
    main()
