"""Resource Inspector for Bravien Training Pipeline.

Inspects GPU model, VRAM, CUDA capability, BF16 support, system RAM, CPU cores,
and disk storage to recommend optimal batching, gradient accumulation, precision,
and offloading strategies.
"""

from __future__ import annotations

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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def inspect_hardware_resources() -> dict[str, Any]:
    """Inspect all system hardware resources and return diagnostic dictionary."""
    # 1. CUDA & GPU
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if cuda_available
        else 0.0
    )
    bf16_supported = torch.cuda.is_bf16_supported() if cuda_available else False
    fp16_supported = cuda_available

    # 2. CPU & RAM
    cpu_count_logical = os.cpu_count() or 1
    cpu_count_physical = (
        psutil.cpu_count(logical=False) if PSUTIL_AVAILABLE else cpu_count_logical // 2
    )
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)
    else:
        total_ram_gb = 16.0
        available_ram_gb = 8.0

    # 3. Disk Space
    disk = shutil.disk_usage(".")
    total_disk_gb = disk.total / (1024**3)
    free_disk_gb = disk.free / (1024**3)

    # 4. Mode Selection
    if cuda_available and total_vram_gb >= 16.0:
        recommended_mode = "full_gpu_fast"
        micro_batch = 4
        grad_accum = 8
        use_cpu_offload = False
    elif cuda_available and total_vram_gb >= 5.5:  # e.g., RTX 4050 6GB
        recommended_mode = "resource_aware_local_gpu"
        micro_batch = 1
        grad_accum = 32
        use_cpu_offload = True
    elif cuda_available:
        recommended_mode = "low_vram_offload"
        micro_batch = 1
        grad_accum = 64
        use_cpu_offload = True
    else:
        recommended_mode = "cpu_simulation"
        micro_batch = 1
        grad_accum = 16
        use_cpu_offload = False

    result = {
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "bf16_supported": bf16_supported,
        "fp16_supported": fp16_supported,
        "cpu_count_logical": cpu_count_logical,
        "cpu_count_physical": cpu_count_physical,
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(available_ram_gb, 2),
        "total_disk_gb": round(total_disk_gb, 2),
        "free_disk_gb": round(free_disk_gb, 2),
        "recommended_training_mode": recommended_mode,
        "recommended_config": {
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": grad_accum,
            "sequence_length": 1024 if total_vram_gb <= 6.0 else 2048,
            "precision": "bf16" if bf16_supported else ("fp16" if fp16_supported else "fp32"),
            "gradient_checkpointing": True,
            "cpu_optimizer_offload": use_cpu_offload,
        },
    }
    return result


def main() -> None:
    resources = inspect_hardware_resources()
    print("=" * 65)
    print("BRAVIEN TRAINING ENVIRONMENT & HARDWARE AUDIT")
    print("=" * 65)
    print(f"CUDA Available:       {resources['cuda_available']}")
    print(f"GPU Model:            {resources['gpu_name']} ({resources['total_vram_gb']} GB VRAM)")
    print(f"BF16 Hardware Native: {resources['bf16_supported']}")
    print(f"System RAM:           {resources['available_ram_gb']} GB free / {resources['total_ram_gb']} GB total")
    print(f"CPU Cores:            {resources['cpu_count_physical']} physical, {resources['cpu_count_logical']} logical")
    print(f"Free Disk Storage:    {resources['free_disk_gb']} GB free / {resources['total_disk_gb']} GB total")
    print("-" * 65)
    print("Recommended Execution Profile:")
    print(f"  Mode:               {resources['recommended_training_mode']}")
    print(f"  Precision:          {resources['recommended_config']['precision']}")
    print(f"  Micro Batch:        {resources['recommended_config']['micro_batch_size']}")
    print(f"  Grad Accumulation:  {resources['recommended_config']['gradient_accumulation_steps']}")
    print(f"  Seq Length:         {resources['recommended_config']['sequence_length']}")
    print(f"  Grad Checkpointing: {resources['recommended_config']['gradient_checkpointing']}")
    print(f"  CPU Optimizer:      {resources['recommended_config']['cpu_optimizer_offload']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
