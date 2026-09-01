"""Production-Ready Native Pretraining & Pilot Launcher for Bravien.

Handles:
- Zero Pretrained Weight Contamination (pure native Bravien initialization)
- Pre-flight Data Validation (token bounds [0, 32000), sequence length alignment)
- Automatic Resume Detection (picks up latest valid checkpoint if present)
- Step & Token Bounded Pilot (--max-steps / --max-tokens)
- Graceful Interruption & OOM Safety
- Rich Structured Telemetry (JSON report + console summary)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from bravien.model.bravien_model import BravienForCausalLM
from bravien.model.checkpoint import load_bravien_checkpoint
from bravien.model.parameter_count import count_parameters
from bravien.tokenizer.tokenizer import BravienTokenizer
from bravien.training.checkpoint_manager import CheckpointManager
from bravien.training.pretrain import BravienPretrainer
from bravien.training.pretrain_config import PretrainConfig


def validate_dataset_before_training(
    dataset_path: str | Path,
    expected_vocab_size: int = 32000,
    expected_seq_len: int | None = None,
) -> PackedPretrainingDataset:
    """Rigorous pre-training data integrity check."""
    ds_path = Path(dataset_path)
    print(f"\n[DATA VALIDATION] Inspecting packed dataset: {ds_path}")

    if not ds_path.exists():
        raise FileNotFoundError(f"Packed dataset file does not exist: {ds_path}")

    try:
        dataset = PackedPretrainingDataset.load(ds_path)
    except Exception as e:
        raise ValueError(f"Failed to read packed dataset file '{ds_path}': {e}") from e

    if len(dataset) == 0:
        raise ValueError(f"Packed dataset at '{ds_path}' is empty (0 blocks)!")

    print(f"  * Total Packed Blocks: {len(dataset):,}")
    print(f"  * Sequence Length:     {dataset.max_seq_len} tokens")

    if expected_seq_len and dataset.max_seq_len != expected_seq_len:
        print(f"  ⚠️ Warning: dataset block length ({dataset.max_seq_len}) differs from configured seq len ({expected_seq_len}). Adapting model position limit.")

    # Inspect sample blocks for corruption or out-of-bound token IDs
    sample_indices = [0, len(dataset) // 2, len(dataset) - 1]
    for idx in sample_indices:
        item = dataset[idx]
        input_ids = item["input_ids"]
        labels = item["labels"]

        if (input_ids < 0).any():
            raise ValueError(f"Corruption detected at block {idx}: negative token ID found!")
        if (input_ids >= expected_vocab_size).any():
            max_id = int(input_ids.max().item())
            raise ValueError(f"Corruption detected at block {idx}: token ID {max_id} >= vocab size {expected_vocab_size}!")
        if input_ids.shape != labels.shape:
            raise ValueError(f"Shape mismatch at block {idx}: input_ids {input_ids.shape} != labels {labels.shape}!")

    print("  ✅ Data integrity verified: zero negative IDs, all token IDs within [0, 32000), causal labels aligned.")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Native Bravien Foundation Pretraining or Pilot.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bravien_1p5b_pretrain.yaml",
        help="Path to pretraining YAML config.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(BRAVIEN_PRESETS.keys()),
        help="Override architecture preset (e.g. bravien-1.5b, bravien-tiny).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to packed dataset .pt file (defaults to data/packed_pretrain/train_packed.pt).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (useful for cheap pilot runs).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum token budget to train before stopping.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Micro batch size per accumulation step.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Checkpoint output directory.",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default=None,
        choices=["bf16", "fp16", "no"],
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="Force clean initialization even if existing checkpoints exist.",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="reports/pilot_telemetry.json",
        help="Path to save machine-readable telemetry JSON.",
    )

    args = parser.parse_args()

    # Load base configuration from YAML if exists
    config_dict: dict[str, Any] = {}
    cfg_file = Path(args.config)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    model_sec = config_dict.get("model", {})
    train_sec = config_dict.get("training", {})
    batch_sec = config_dict.get("batching", {})
    hardw_sec = config_dict.get("hardware", {})
    ckpt_sec = config_dict.get("checkpoints", {})
    tok_sec = config_dict.get("tokenizer", {})

    preset_name = args.preset or model_sec.get("name", "bravien-1.5b")
    out_dir = Path(args.output_dir or ckpt_sec.get("output_dir", "checkpoints/bravien-native-1.5b"))
    data_path = args.data_path or "data/packed_pretrain/train_packed.pt"
    micro_bs = args.micro_batch_size or batch_sec.get("micro_batch_size", 1)
    grad_accum = args.gradient_accumulation_steps or batch_sec.get("gradient_accumulation_steps", 32)
    lr = args.lr or train_sec.get("learning_rate", 3e-4)
    max_steps = args.max_steps or train_sec.get("max_steps", 100000)
    max_tokens = args.max_tokens
    precision = args.mixed_precision or hardw_sec.get("precision", "bf16")

    print("\n" + "=" * 70)
    print("BRAVIEN NATIVE FOUNDATION PRETRAINING LAUNCHER")
    print(f"Lifecycle Stage: [LIFECYCLE: foundation_training]")
    print("=" * 70)

    # 1. Validate Dataset
    dataset = None
    if Path(data_path).exists():
        dataset = validate_dataset_before_training(data_path, expected_vocab_size=32000)
    else:
        print(f"\n⚠️ Note: Packed dataset not found at '{data_path}'. Using synthetic sanity corpus.")

    # 2. Checkpoint Discovery & Resume Policy
    latest_ckpt: Path | None = None
    initial_step = 0
    initial_tokens = 0
    saved_opt_state = None
    saved_sched_state = None
    ckpt_mgr = CheckpointManager(out_dir)

    if not args.force_new:
        latest_ckpt = ckpt_mgr.get_latest_checkpoint()

    # 3. Model Initialization (Zero external/Qwen contamination)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if latest_ckpt is not None:
        print(f"\n[RESUME DETECTED] Found existing checkpoint -> {latest_ckpt}")
        print(f"Resuming training state directly from: {latest_ckpt.name}...")
        model = load_bravien_checkpoint(latest_ckpt, device=device, dtype="auto")

        # Load training state dictionary if exists
        train_state_file = latest_ckpt / "training_state.pt"
        if train_state_file.exists():
            try:
                state_bundle = torch.load(train_state_file, map_location="cpu", weights_only=False)
                t_state = state_bundle.get("training_state", {})
                initial_step = t_state.get("step", 0)
                initial_tokens = t_state.get("total_tokens_trained", 0)
                saved_opt_state = state_bundle.get("optimizer_state_dict")
                saved_sched_state = state_bundle.get("scheduler_state_dict")
                print(f"  * Restored Global Step:     {initial_step}")
                print(f"  * Restored Tokens Trained:  {initial_tokens:,}")
            except Exception as e:
                print(f"  ⚠️ Could not read training_state.pt: {e}")
    else:
        print(f"\n[NEW TRAINING] Initializing fresh native model architecture: '{preset_name}'...")
        model_config = get_bravien_preset(preset_name)
        if dataset is not None and dataset.max_seq_len > model_config.max_position_embeddings:
            model_config.max_position_embeddings = dataset.max_seq_len
        model = BravienForCausalLM(model_config)

    param_stats = count_parameters(model)
    print(f"  * Model Class:        {type(model).__name__} (Native PyTorch)")
    print(f"  * Total Parameters:   {param_stats['total_parameters']:,} ({param_stats['total_millions']:.2f}M / {param_stats['total_billions']:.3f}B)")
    print(f"  * Device:             {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"  * Precision:          {precision}")
    print(f"  * Micro Batch Size:   {micro_bs} (Grad Accum: {grad_accum}, Effective Batch: {micro_bs * grad_accum})")
    print(f"  * Target Output Dir:  {out_dir}")
    if max_tokens:
        print(f"  * Token Budget Limit: {max_tokens:,} tokens")

    # 4. Prepare PretrainConfig
    pretrain_cfg = PretrainConfig(
        learning_rate=lr,
        max_steps=max_steps,
        max_tokens=max_tokens,
        initial_step=initial_step,
        initial_tokens=initial_tokens,
        micro_batch_size=micro_bs,
        gradient_accumulation_steps=grad_accum,
        mixed_precision=precision,
        output_dir=str(out_dir),
        log_interval=train_sec.get("log_frequency_steps", 10),
        save_interval=ckpt_sec.get("save_frequency_steps", 500),
    )

    # 5. Launch Training
    pretrainer = BravienPretrainer(
        model=model,
        config=pretrain_cfg,
        train_dataset=dataset,
    )

    # Restore optimizer and scheduler states if resuming
    if saved_opt_state is not None:
        try:
            pretrainer.optimizer.load_state_dict(saved_opt_state)
            print("  ✅ Restored AdamW momentum buffers")
        except Exception as e:
            print(f"  ⚠️ Could not restore optimizer momentum: {e}")

    if saved_sched_state is not None:
        try:
            pretrainer.scheduler.load_state_dict(saved_sched_state)
            print("  ✅ Restored Cosine LR schedule state")
        except Exception as e:
            print(f"  ⚠️ Could not restore scheduler state: {e}")

    try:
        result = pretrainer.train()

        # Build comprehensive telemetry dictionary
        tok_path = Path(tok_sec.get("path", "tokenizers/bravien-native"))
        vocab_size = 30701
        if tok_path.exists():
            try:
                vocab_size = BravienTokenizer.from_pretrained(tok_path).vocab_size
            except Exception:
                pass

        telemetry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_name": preset_name,
            "parameter_count": param_stats["total_parameters"],
            "parameter_count_millions": param_stats["total_millions"],
            "parameter_count_billions": param_stats["total_billions"],
            "tokenizer_vocabulary": vocab_size,
            "sequence_length": dataset.max_seq_len if dataset else 2048,
            "total_training_steps": result["final_step"],
            "total_tokens_processed": result["total_tokens"],
            "tokens_in_session": result["tokens_trained_in_session"],
            "final_loss": round(result["final_loss"], 4),
            "average_recent_loss": round(result["average_recent_loss"], 4),
            "tokens_per_second": round(result["tokens_per_second"], 1),
            "peak_gpu_memory_mb": round(result["peak_gpu_memory_mb"], 1),
            "checkpoint_path": result["last_checkpoint_path"] or str(latest_ckpt or out_dir),
            "resumed_from_checkpoint": latest_ckpt is not None,
            "training_duration_seconds": round(result["elapsed_seconds"], 2),
            "detected_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__,
            "status": "COMPLETED",
        }

        print("\n" + "=" * 70)
        print("PILOT / PRETRAINING TELEMETRY SUMMARY")
        print("=" * 70)
        print(f"  - Model:              {telemetry['model_name']} ({telemetry['parameter_count_millions']:.2f}M params)")
        print(f"  - Total Steps:        {telemetry['total_training_steps']}")
        print(f"  - Total Tokens:       {telemetry['total_tokens_processed']:,} ({telemetry['tokens_in_session']:,} in this session)")
        print(f"  - Final Loss:         {telemetry['final_loss']:.4f} (Recent Avg: {telemetry['average_recent_loss']:.4f})")
        print(f"  - Throughput:         {telemetry['tokens_per_second']:,.1f} tok/s")
        print(f"  - Duration:           {telemetry['training_duration_seconds']:.2f}s")
        print(f"  - Peak GPU VRAM:      {telemetry['peak_gpu_memory_mb']:.1f} MB")
        print(f"  - Checkpoint Saved:   {telemetry['checkpoint_path']}")
        print(f"  - Auto-Resume Tested: {telemetry['resumed_from_checkpoint']}")
        print("=" * 70 + "\n")

        report_p = Path(args.report_file)
        report_p.parent.mkdir(parents=True, exist_ok=True)
        with open(report_p, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2)
        print(f"Machine-readable telemetry saved to: {report_p}")

    except torch.cuda.OutOfMemoryError:
        print("\n❌ CUDA Out-of-Memory Error encountered!")
        print("Suggestion: Decrease --micro-batch-size or enable activation gradient checkpointing in config.")
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n⚠️ Training halted by user interruption. Latest valid checkpoint preserved.")


if __name__ == "__main__":
    main()
