"""Bravien Modal Cloud Pretraining & Control Layer.

Mounts the Bravien codebase into a Debian-slim Linux container with PyTorch,
binds a persistent Modal Volume at /mnt/bravien, and executes native Bravien
preflight and training scripts remotely with explicit GPU tier and cost safety.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# -----------------------------------------------------------------------------
# 1. Project Root & Git Metadata
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()

def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"

GIT_COMMIT = get_git_commit()

# -----------------------------------------------------------------------------
# 2. Modal App & Persistent Storage Volume
# -----------------------------------------------------------------------------
app = modal.App("bravien-pretraining")

# Persistent volume mounted at /mnt/bravien
# Structure:
# /mnt/bravien/
#   ├── checkpoints/
#   ├── reports/
#   ├── logs/
#   └── data/
volume = modal.Volume.from_name("bravien-storage", create_if_missing=True)

# -----------------------------------------------------------------------------
# 3. Base Container Image with Dependencies & Code Mount
# -----------------------------------------------------------------------------
bravien_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "pyyaml",
        "pydantic",
    )
    .add_local_dir(REPO_ROOT / "bravien", remote_path="/root/Bravien/bravien", copy=True)
    .add_local_dir(REPO_ROOT / "configs", remote_path="/root/Bravien/configs", copy=True)
    .add_local_dir(REPO_ROOT / "tokenizers", remote_path="/root/Bravien/tokenizers", copy=True)
    .add_local_dir(REPO_ROOT / "scripts", remote_path="/root/Bravien/scripts", copy=True)
    .add_local_dir(REPO_ROOT / "manifests", remote_path="/root/Bravien/manifests", copy=True)
    .env({"BRAVIEN_GIT_COMMIT": GIT_COMMIT})
)

# -----------------------------------------------------------------------------
# 4. Remote Control Functions
# -----------------------------------------------------------------------------

@app.function(
    image=bravien_image,
    volumes={"/mnt/bravien": volume},
    timeout=120,
)
def remote_health_check() -> dict:
    """Zero-GPU lightweight CPU verification of remote container & persistent volume."""
    import platform
    import torch

    # Ensure persistent directories exist inside volume
    base_vol = Path("/mnt/bravien")
    (base_vol / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_vol / "reports").mkdir(parents=True, exist_ok=True)
    (base_vol / "logs").mkdir(parents=True, exist_ok=True)
    (base_vol / "data").mkdir(parents=True, exist_ok=True)
    volume.commit()

    return {
        "status": "HEALTHY",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": os.environ.get("BRAVIEN_GIT_COMMIT", "unknown"),
        "volume_mount": str(base_vol),
        "volume_subdirs": [p.name for p in base_vol.iterdir() if p.is_dir()],
    }


@app.function(
    image=bravien_image,
    gpu="T4",  # Default to cheapest available GPU tier
    volumes={"/mnt/bravien": volume},
    timeout=300,
)
def remote_preflight(config_path: str = "configs/bravien_1p5b_pretrain.yaml") -> dict:
    """Run preflight environmental and configuration audit on remote GPU instance."""
    import sys
    sys.path.insert(0, "/root/Bravien")

    from scripts.preflight_pretraining import run_preflight
    report = run_preflight(Path("/root/Bravien") / config_path)
    return report


@app.function(
    image=bravien_image,
    gpu="T4",  # Cheapest GPU tier for smoke verification
    volumes={"/mnt/bravien": volume},
    timeout=600,
)
def remote_smoke_test() -> dict:
    """Run lightweight 10-step remote GPU smoke test on bravien-tiny preset."""
    cmd = [
        sys.executable,
        "-u",
        "/root/Bravien/scripts/pretrain_bravien.py",
        "--preset", "bravien-tiny",
        "--max-steps", "10",
        "--output-dir", "/mnt/bravien/checkpoints/smoke_test",
        "--report-file", "/mnt/bravien/reports/smoke_telemetry.json",
    ]
    res = subprocess.run(cmd, cwd="/root/Bravien", capture_output=True, text=True)
    volume.commit()

    return {
        "returncode": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }


@app.function(
    image=bravien_image,
    volumes={"/mnt/bravien": volume},
    timeout=3600,
)
def remote_train_job(
    config_path: str = "configs/bravien_1p5b_pretrain.yaml",
    data_path: str = "/mnt/bravien/data/train_packed.pt",
    max_steps: int | None = 50,
    max_tokens: int | None = None,
    output_dir: str = "/mnt/bravien/checkpoints/bravien-pilot",
) -> dict:
    """Run bounded foundation pretraining pilot on remote GPU instance."""
    cmd = [
        sys.executable,
        "-u",
        "/root/Bravien/scripts/pretrain_bravien.py",
        "--config", f"/root/Bravien/{config_path}",
        "--data-path", data_path,
        "--output-dir", output_dir,
        "--report-file", f"{output_dir}/telemetry.json",
    ]
    if max_steps:
        cmd.extend(["--max-steps", str(max_steps)])
    if max_tokens:
        cmd.extend(["--max-tokens", str(max_tokens)])

    res = subprocess.run(cmd, cwd="/root/Bravien", capture_output=True, text=True)
    volume.commit()

    return {
        "returncode": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }
