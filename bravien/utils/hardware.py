"""Device selection and hardware reporting (§20, §58).

Bravien must run on whatever the user has. This module answers "what am I
running on?" honestly — it reports what it detected, never what it hopes for.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DeviceInfo:
    """What Bravien found, and what it will therefore do."""

    device: str
    backend: str
    name: str = ""
    total_memory_gb: float = 0.0
    compute_capability: tuple[int, int] | None = None
    supports_bf16: bool = False
    supports_fp16: bool = False
    cuda_version: str | None = None
    torch_version: str = ""
    cpu_count: int = 0
    system_memory_gb: float = 0.0
    platform: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    @property
    def preferred_dtype(self) -> torch.dtype:
        """The widest-range dtype this device can train in without loss scaling.

        BF16 has FP32's exponent range, so it needs no GradScaler. FP16 does,
        which is why it is only chosen when BF16 is unavailable.
        """
        if self.supports_bf16:
            return torch.bfloat16
        if self.supports_fp16 and self.is_cuda:
            return torch.float16
        return torch.float32

    @property
    def needs_grad_scaler(self) -> bool:
        return self.preferred_dtype == torch.float16

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "backend": self.backend,
            "name": self.name,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "compute_capability": (
                f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"
                if self.compute_capability
                else None
            ),
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "preferred_dtype": str(self.preferred_dtype).replace("torch.", ""),
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "cpu_count": self.cpu_count,
            "system_memory_gb": round(self.system_memory_gb, 2),
            "platform": self.platform,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        lines = [
            "Bravien hardware",
            "",
            f"  Device:      {self.device}  ({self.backend})",
        ]
        if self.name:
            lines.append(f"  Name:        {self.name}")
        if self.total_memory_gb:
            lines.append(f"  Memory:      {self.total_memory_gb:.2f} GB")
        if self.compute_capability:
            major, minor = self.compute_capability
            lines.append(f"  Capability:  sm_{major}{minor}")
        lines += [
            f"  Precision:   bf16={self.supports_bf16}  fp16={self.supports_fp16}"
            f"  -> {str(self.preferred_dtype).replace('torch.', '')}",
            f"  Torch:       {self.torch_version}"
            + (f"  (CUDA {self.cuda_version})" if self.cuda_version else ""),
            f"  Host:        {self.platform}, {self.cpu_count} CPU threads,"
            f" {self.system_memory_gb:.1f} GB RAM",
        ]
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)


def _system_memory_gb() -> float:
    """Total RAM, using only the standard library.

    `os.sysconf` covers Linux and macOS; Windows needs a ctypes call. Returns 0.0
    when it cannot be determined rather than guessing.
    """
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / 1024**3
    except (OSError, ValueError, KeyError):
        pass

    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / 1024**3
        except (ImportError, AttributeError, OSError):
            pass
    return 0.0


def detect_device(requested: str | None = None) -> DeviceInfo:
    """Pick a device, falling back CUDA -> MPS -> CPU (§58).

    Args:
        requested: force a device ("cuda", "cuda:1", "mps", "cpu"). If the
            request cannot be honoured, this falls back and records *why* in
            `notes` rather than failing or pretending.
    """
    info = DeviceInfo(
        device="cpu",
        backend="cpu",
        torch_version=torch.__version__,
        cpu_count=os.cpu_count() or 1,
        system_memory_gb=_system_memory_gb(),
        platform=platform.platform(),
    )

    requested = (requested or os.environ.get("BRAVIEN_DEVICE") or "auto").lower()

    want_cuda = requested.startswith("cuda") or requested == "auto"
    want_mps = requested == "mps" or requested == "auto"

    if want_cuda and torch.cuda.is_available():
        index = 0
        if ":" in requested:
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError:
                info.notes.append(f"could not parse device index in {requested!r}")
        if index >= torch.cuda.device_count():
            info.notes.append(
                f"cuda:{index} requested but only {torch.cuda.device_count()} "
                f"device(s) present; using cuda:0"
            )
            index = 0

        props = torch.cuda.get_device_properties(index)
        info.device = f"cuda:{index}"
        info.backend = "cuda"
        info.name = props.name
        info.total_memory_gb = props.total_memory / 1024**3
        info.compute_capability = (props.major, props.minor)
        info.supports_bf16 = torch.cuda.is_bf16_supported()
        info.supports_fp16 = True
        info.cuda_version = torch.version.cuda
        if not info.supports_bf16:
            info.notes.append(
                "bf16 unsupported on this GPU; fp16 with gradient scaling will "
                "be used, which is less numerically forgiving"
            )
        return info

    if requested.startswith("cuda") and not torch.cuda.is_available():
        info.notes.append(
            "CUDA requested but torch.cuda.is_available() is False — check the "
            "driver and that torch was installed with a CUDA build"
        )

    mps = getattr(torch.backends, "mps", None)
    if want_mps and mps is not None and mps.is_available():
        info.device = "mps"
        info.backend = "mps"
        info.name = "Apple Silicon (Metal)"
        info.supports_bf16 = False
        info.supports_fp16 = True
        info.total_memory_gb = info.system_memory_gb  # unified memory
        info.notes.append("MPS: unified memory is shared with the OS")
        return info

    if requested == "mps":
        info.notes.append("MPS requested but unavailable; falling back to CPU")

    # CPU: bf16 works on any recent PyTorch but is slow without AVX512-BF16.
    info.name = platform.processor() or "CPU"
    info.supports_bf16 = False
    info.supports_fp16 = False
    if requested == "auto":
        info.notes.append(
            "no GPU found — training will be slow; use the 'tiny' preset"
        )
    return info


def resolve_device(requested: str | None = None) -> torch.device:
    """Just the device, for callers that do not need the full report."""
    return torch.device(detect_device(requested).device)


def disk_free_gb(path: str = ".") -> float:
    return shutil.disk_usage(path).free / 1024**3
