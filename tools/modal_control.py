"""Deterministic Local Control CLI for Bravien Modal Operations.

Provides reliable execution of:
- status:    Local Modal CLI + authenticated profile + remote CPU container health check
- preflight: Remote GPU environment & configuration audit
- smoke:     Tiny 10-step remote GPU smoke test
- pilot:     Bounded real-data pilot (--steps N or --tokens N)
- resume:    Automatic resumption of previous training on Modal Volume

Enforces Cost Safety:
- Prints GPU tier and estimated cost before execution
- Defaults to the cheapest available GPU tier (T4 / L4)
- Prohibits automated selection of A100 / H100 / multi-GPU
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import modal

# Import Modal App and functions from modal/bravien_modal.py
sys.path.insert(0, str(REPO_ROOT / "modal"))
from bravien_modal import (
    app,
    get_git_commit,
    remote_health_check,
    remote_preflight,
    remote_smoke_test,
    remote_train_job,
)

GPU_COST_GUIDE = {
    "T4": "$0.59 / hr (Entry GPU - Best for preflight & smoke)",
    "L4": "$0.80 / hr (Ada Lovelace, 24GB - Best value for BF16 pilot)",
    "A10G": "$1.10 / hr (Ampere, 24GB - High throughput pilot)",
}


def print_cost_safety_banner(gpu_tier: str) -> None:
    print("\n" + "=" * 65)
    print("MODAL COST & RESOURCE SAFETY")
    print(f"Target GPU Tier:       {gpu_tier}")
    print(f"Estimated Cost Rate:   {GPU_COST_GUIDE.get(gpu_tier, 'Custom tier')}")
    print(f"Safety Rule:           Zero expensive tiers (A100/H100) auto-selected")
    print(f"Git Commit:            {get_git_commit()}")
    print("=" * 65 + "\n")


def cmd_status() -> None:
    print("=" * 65)
    print("BRAVIEN MODAL CONTROL: WORKSPACE & HEALTH STATUS")
    print("=" * 65)

    # 1. Local CLI Check
    cli_version = modal.__version__
    print(f"  * Modal SDK Version:       {cli_version}")

    # 2. Local Profile Check
    try:
        res = subprocess.run(
            [sys.executable, "-m", "modal", "profile", "current"],
            capture_output=True,
            text=True,
            check=True,
        )
        profile_name = res.stdout.strip().splitlines()[-1]
        print(f"  * Authenticated Profile:   {profile_name}")
    except Exception as e:
        print(f"  ❌ Could not detect profile: {e}")
        return

    # 3. Remote CPU Health Check (Zero GPU cost)
    print("\nExecuting remote zero-GPU container health check on Modal...")
    with app.run():
        health = remote_health_check.remote()

    print(f"  ✅ Remote Container Status: {health['status']}")
    print(f"  * Remote Python Version:   {health['python_version']}")
    print(f"  * Remote Platform:         {health['platform']}")
    print(f"  * Remote PyTorch Version:  {health['torch_version']}")
    print(f"  * Remote CUDA Available:   {health['cuda_available']}")
    print(f"  * Remote Commit:           {health['git_commit']}")
    print(f"  * Persistent Volume Mount: {health['volume_mount']}")
    print(f"  * Volume Subdirectories:   {', '.join(health['volume_subdirs'])}")
    print("\n[READY] Modal workspace is fully reachable and responsive.\n")


def cmd_preflight(gpu: str, config: str) -> None:
    print_cost_safety_banner(gpu)
    print(f"Running remote GPU preflight audit on Modal [{gpu}]...")

    with app.run():
        report = remote_preflight.remote(config_path=config)

    print(json.dumps(report, indent=2))
    if report.get("status") == "PASS":
        print("\n✅ Remote GPU Preflight PASSED.")
    else:
        print("\n❌ Remote GPU Preflight FAILED.")


def cmd_smoke(gpu: str) -> None:
    print_cost_safety_banner(gpu)
    print(f"Running lightweight 10-step remote GPU smoke test on Modal [{gpu}]...")

    with app.run():
        result = remote_smoke_test.remote()

    print(result["stdout"])
    if result["stderr"]:
        print("STDERR:", result["stderr"])

    if result["returncode"] == 0:
        print("\n✅ Remote Smoke Test COMPLETED successfully.")
    else:
        print(f"\n❌ Remote Smoke Test failed with exit code {result['returncode']}.")


def cmd_pilot(
    steps: int | None,
    tokens: int | None,
    gpu: str,
    config: str,
    data_path: str,
    output_dir: str,
) -> None:
    print_cost_safety_banner(gpu)
    limit_desc = f"{tokens:,} tokens" if tokens else f"{steps} steps"
    print(f"Launching bounded real-data pilot ({limit_desc}) on Modal [{gpu}]...")

    with app.run():
        result = remote_train_job.remote(
            config_path=config,
            data_path=data_path,
            max_steps=steps,
            max_tokens=tokens,
            output_dir=output_dir,
        )

    print(result["stdout"])
    if result["stderr"]:
        print("STDERR:", result["stderr"])

    if result["returncode"] == 0:
        print("\n✅ Remote Pilot Run COMPLETED successfully.")
    else:
        print(f"\n❌ Remote Pilot Run failed with exit code {result['returncode']}.")


def cmd_resume(gpu: str, config: str, output_dir: str, data_path: str) -> None:
    print_cost_safety_banner(gpu)
    print(f"Resuming training run from Modal Volume ({output_dir}) on Modal [{gpu}]...")

    with app.run():
        result = remote_train_job.remote(
            config_path=config,
            data_path=data_path,
            output_dir=output_dir,
        )

    print(result["stdout"])
    if result["stderr"]:
        print("STDERR:", result["stderr"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Bravien Modal Control CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    subparsers.add_parser("status", help="Check local Modal auth and run zero-GPU remote health check.")

    # preflight
    preflight_parser = subparsers.add_parser("preflight", help="Run remote preflight check on Modal GPU.")
    preflight_parser.add_argument("--gpu", type=str, default="T4", choices=["T4", "L4", "A10G"], help="GPU tier.")
    preflight_parser.add_argument("--config", type=str, default="configs/bravien_1p5b_pretrain.yaml")

    # smoke
    smoke_parser = subparsers.add_parser("smoke", help="Run 10-step remote GPU smoke test.")
    smoke_parser.add_argument("--gpu", type=str, default="T4", choices=["T4", "L4", "A10G"], help="GPU tier.")

    # pilot
    pilot_parser = subparsers.add_parser("pilot", help="Run bounded real-data pilot on Modal GPU.")
    pilot_parser.add_argument("--steps", type=int, default=50, help="Maximum steps.")
    pilot_parser.add_argument("--tokens", type=int, default=None, help="Maximum token budget.")
    pilot_parser.add_argument("--gpu", type=str, default="L4", choices=["T4", "L4", "A10G"], help="GPU tier.")
    pilot_parser.add_argument("--config", type=str, default="configs/bravien_1p5b_pretrain.yaml")
    pilot_parser.add_argument("--data-path", type=str, default="/mnt/bravien/data/train_packed.pt")
    pilot_parser.add_argument("--output-dir", type=str, default="/mnt/bravien/checkpoints/bravien-pilot")

    # resume
    resume_parser = subparsers.add_parser("resume", help="Resume training from latest checkpoint on volume.")
    resume_parser.add_argument("--gpu", type=str, default="L4", choices=["T4", "L4", "A10G"], help="GPU tier.")
    resume_parser.add_argument("--config", type=str, default="configs/bravien_1p5b_pretrain.yaml")
    resume_parser.add_argument("--data-path", type=str, default="/mnt/bravien/data/train_packed.pt")
    resume_parser.add_argument("--output-dir", type=str, default="/mnt/bravien/checkpoints/bravien-pilot")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "preflight":
        cmd_preflight(args.gpu, args.config)
    elif args.command == "smoke":
        cmd_smoke(args.gpu)
    elif args.command == "pilot":
        cmd_pilot(args.steps, args.tokens, args.gpu, args.config, args.data_path, args.output_dir)
    elif args.command == "resume":
        cmd_resume(args.gpu, args.config, args.output_dir, args.data_path)


if __name__ == "__main__":
    main()
