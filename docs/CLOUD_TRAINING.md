# Bravien Cloud & Remote GPU Training Guide

This guide details the setup, environment requirements, operational workflows, and execution commands for foundation pretraining, Supervised Fine-Tuning (SFT), and Direct Preference Optimization (DPO) for the **Bravien-1.5B** native model architecture on remote cloud GPUs (e.g., AWS, GCP, Lambda Labs, RunPod, Kaggle, Google Colab).

---

## 1. System Requirements

### Hardware Requirements
- **Recommended GPU**: NVIDIA A100 (40GB/80GB), H100 (80GB), or L40S.
- **Minimum GPU (Single Node)**: NVIDIA RTX 4090 (24GB) or A10G (24GB) with activation gradient checkpointing enabled.
- **RAM**: Minimum 32 GB System RAM (64 GB+ recommended for dataset packing and tokenizer processing).
- **Storage**: Fast NVMe SSD with 100 GB+ free disk space for packed tokens, checkpoints, and telemetry logs.

### Software Stack
- **OS**: Linux (Ubuntu 22.04 LTS or 24.04 LTS recommended).
- **CUDA**: 12.1 or higher.
- **Python**: 3.10, 3.11, or 3.12.
- **PyTorch**: 2.2+ with CUDA compute capability (`torch>=2.2.0`).
- **Precision**: Native `bfloat16` support (Compute Capability $\ge 8.0$: Ampere, Ada Lovelace, Hopper).

---

## 2. Directory Structure & Staging Isolation

To maintain strict separation between SFT data and foundation pretraining corpus:

```
Bravien/
├── data/
│   ├── raw_foundation/        # Staged raw foundation JSONL / text files (gitignored)
│   ├── processed_foundation/  # Cleaned / deduplicated foundation data (gitignored)
│   ├── packed_pretrain/       # Contiguous 4096-token packed .pt blocks (gitignored)
│   ├── processed/             # SFT and multi-turn instruction datasets (tracked manifests)
│   └── manifests/             # Dataset provenance & statistics manifests
├── tokenizers/
│   └── bravien-native/        # Verified 30,701-vocabulary native BPE tokenizer (tracked)
├── configs/
│   └── bravien_1p5b_pretrain.yaml # Canonical pretraining configuration
└── checkpoints/
    └── bravien-native-1.5b/   # Rolling step_XXXXXXX checkpoint directories
```

---

## 3. Exact Operational Commands

### COMMAND A: Cloud Environment Preflight
Inspects GPU VRAM, CUDA, PyTorch, BF16 capabilities, tokenizer, and config **without starting training**:

```bash
python scripts/preflight_pretraining.py --config configs/bravien_1p5b_pretrain.yaml
```

### COMMAND B: Cheap Real-Data Pilot
Executes a fast, bounded verification run on real packed tokens to validate the full pipeline (data loading, forward pass, loss, backpropagation, optimizer update, and atomic checkpointing) before committing to a full run:

```bash
python scripts/pretrain_bravien.py \
    --config configs/bravien_1p5b_pretrain.yaml \
    --data-path data/packed_pretrain/train_packed.pt \
    --max-steps 50 \
    --output-dir checkpoints/bravien-pilot
```

### COMMAND C: Full Foundation Pretraining
Launches full-scale pretraining on the staged foundation dataset with automatic resume detection and rolling checkpoint protection:

```bash
python scripts/pretrain_bravien.py \
    --config configs/bravien_1p5b_pretrain.yaml \
    --data-path data/packed_pretrain/bravien_1p5b_pretrain.pt \
    --output-dir checkpoints/bravien-native-1.5b
```

---

## 4. Pretraining Data Preparation & Packing

Pack raw texts into fixed-length contiguous token blocks (zero padding waste, causal EOS delimiters):

```bash
python scripts/pack_pretraining_data.py \
    --data-dir data/raw_foundation \
    --tokenizer-dir tokenizers/bravien-native \
    --output-file data/packed_pretrain/bravien_1p5b_pretrain.pt \
    --seq-len 4096
```

---

## 5. Checkpoint & Resume Policy

- **Atomic Writes**: Checkpoints write to `temp_step_*` and rename atomically to `step_*` to prevent partial corruption during spot instance preemption.
- **Automatic Resume**: If valid checkpoints exist in `--output-dir`, `pretrain_bravien.py` automatically detects and restores the latest step, optimizer momentum buffers, learning rate schedule, and token counter.
- **Zero Contamination**: Initializing a new model builds purely from native `BravienForCausalLM` random initialization. Zero weights are downloaded from Qwen, Llama, or external models.

---

## 6. SFT & Alignment Workflows

### Supervised Fine-Tuning (SFT)
```bash
python scripts/train_bravien.py \
    --model-checkpoint checkpoints/bravien-native-1.5b/step_0100000 \
    --data-path data/processed/train.jsonl \
    --output-dir checkpoints/bravien-1.5b-sft \
    --epochs 3 \
    --lr 2e-5
```

### Direct Preference Optimization (DPO)
```bash
python scripts/align_bravien.py \
    --model-checkpoint checkpoints/bravien-1.5b-sft/final \
    --preference-data data/processed/dpo_preferences.jsonl \
    --output-dir checkpoints/bravien-v4-aligned \
    --beta 0.1 \
    --lr 5e-6
```
