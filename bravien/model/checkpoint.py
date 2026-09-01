"""Native Bravien Checkpoint Management and Serialization.

Provides robust, self-contained checkpoint loading, saving, and cryptographic
SHA-256 manifest verification without external framework dependencies.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from bravien.model.bravien_config import BravienConfig


@dataclass
class CheckpointManifest:
    """Cryptographic manifest and metadata for a Bravien model checkpoint."""

    model_name: str
    architecture_version: str
    created_at: str
    total_parameters: int
    trainable_parameters: int
    vocab_size: int
    hidden_size: int
    num_layers: int
    files: dict[str, str] = field(default_factory=dict)  # filename -> sha256
    training_steps: int = 0
    training_loss: float = 0.0
    tokenizer_version: str = "1.0.0"
    license: str = "Apache-2.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointManifest:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _compute_sha256(filepath: Path) -> str:
    """Compute hex SHA-256 digest of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def save_bravien_checkpoint(
    model: Any,  # BravienForCausalLM
    save_directory: str | Path,
    manifest_metadata: dict[str, Any] | None = None,
    training_steps: int = 0,
    training_loss: float = 0.0,
) -> Path:
    """Save a native Bravien checkpoint with model config, weights, and SHA-256 manifest."""
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Config
    config_path = save_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, indent=2)

    # 2. Save Weights
    weights_path = save_dir / "model.pt"
    # Filter state dict for tied weights
    state_dict = model.state_dict()
    torch.save(state_dict, weights_path)

    # 3. Compute Checksums & Save Manifest
    param_report = model.count_parameters()
    files_checksums = {
        "model_config.json": _compute_sha256(config_path),
        "model.pt": _compute_sha256(weights_path),
    }

    manifest = CheckpointManifest(
        model_name=model.config.name,
        architecture_version=model.config.architecture_version,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_parameters=param_report.total,
        trainable_parameters=param_report.trainable,
        vocab_size=model.config.vocab_size,
        hidden_size=model.config.hidden_size,
        num_layers=model.config.num_layers,
        files=files_checksums,
        training_steps=training_steps,
        training_loss=training_loss,
        metadata=manifest_metadata or {},
    )

    manifest_path = save_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return save_dir


def load_bravien_checkpoint(
    save_directory: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | str = "auto",
    verify_checksums: bool = False,
) -> Any:
    """Load a native Bravien checkpoint into a BravienForCausalLM instance."""
    from bravien.model.bravien_model import BravienForCausalLM

    load_dir = Path(save_directory)
    if not load_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {load_dir}")

    config_path = load_dir / "model_config.json"
    if not config_path.exists():
        config_path = load_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No model_config.json found in {load_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = BravienConfig.from_dict(config_dict)

    # Optional SHA-256 integrity verification
    manifest_path = load_dir / "manifest.json"
    if verify_checksums and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = CheckpointManifest.from_dict(json.load(f))
        for filename, expected_hash in manifest.files.items():
            file_path = load_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Manifest expects '{filename}' but file is missing")
            actual_hash = _compute_sha256(file_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Integrity check failed for {filename}: expected {expected_hash}, got {actual_hash}"
                )

    # Cast dtype target
    target_dtype = getattr(torch, dtype) if isinstance(dtype, str) and dtype != "auto" else (dtype if dtype != "auto" else torch.float32)

    # Instantiate model in target dtype
    model = BravienForCausalLM(config).to(dtype=target_dtype)

    # Load weights
    weights_path = load_dir / "model.pt"
    if not weights_path.exists():
        weights_path = load_dir / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"No model weights found in {load_dir}")

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if target_dtype != torch.float32:
        for k, v in list(state_dict.items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                state_dict[k] = v.to(dtype=target_dtype)

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    import gc
    gc.collect()

    model.to(device=device)
    model.eval()
    return model
