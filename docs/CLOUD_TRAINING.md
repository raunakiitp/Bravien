# Bravien Cloud & Remote GPU Training Guide

This guide details the setup, environment requirements, and operational workflows for executing large-scale foundation pretraining, Supervised Fine-Tuning (SFT), and Direct Preference Optimization (DPO) for the **Bravien-1.5B** native model architecture on remote cloud GPUs (e.g., AWS, GCP, Lambda Labs, RunPod, Kaggle, Google Colab).

---

## 1. System Requirements

### Hardware Requirements
- **Recommended GPU**: NVIDIA A100 (40GB/80GB), H100 (80GB), or L40S.
- **Minimum GPU (Single Node)**: NVIDIA RTX 4090 (24GB) or A10G (24GB) with activation checkpointing enabled.
- **RAM**: Minimum 32 GB System RAM (64 GB+ recommended for dataset loading and tokenizer processing).
- **Storage**: Fast NVMe SSD with 100 GB+ free disk space for packed tokens, checkpoints, and telemetry logs.

### Software Stack
- **OS**: Linux (Ubuntu 22.04 LTS or 24.04 LTS recommended).
- **CUDA**: 12.1 or higher.
- **Python**: 3.10, 3.11, or 3.12.
- **PyTorch**: 2.2+ with CUDA compute capability (`torch>=2.2.0`).
- **Precision**: Native `bfloat16` support (Compute Capability $\ge 8.0$: Ampere, Ada Lovelace, Hopper).

---

## 2. Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/raunakiitp/Bravien.git
cd Bravien

# 2. Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install Bravien dependencies
pip install -e .
pip install transformers accelerate datasets wandb fastapi uvicorn pydantic pyyaml
```

---

## 3. Training Execution Workflow

### A. Pretraining Data Preparation & Packing
Pack tokenized sequences into contiguous 4096-token context chunks with `<EOS>` delimiters for zero padding waste:

```bash
python scripts/pack_pretraining_data.py \
    --input data/processed/train.jsonl \
    --output data/packed_pretrain/bravien_pretrain.pt \
    --tokenizer tokenizers/bravien-native \
    --seq-len 4096
```

### B. Launching Distributed / Single-GPU Pretraining
Run pretraining using the verified configuration [`configs/bravien_1p5b_pretrain.yaml`](file:///c:/Users/LOQ/OneDrive/Documents/Bravien/configs/bravien_1p5b_pretrain.yaml):

```bash
python scripts/resume_pretraining.py \
    --config configs/bravien_1p5b_pretrain.yaml \
    --data-path data/packed_pretrain/bravien_pretrain.pt \
    --output-dir checkpoints/bravien-1.5b-pretrain \
    --dtype bfloat16 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 3e-4 \
    --save-every 1000 \
    --log-every 20
```

### C. Resuming from Interruptions
The pretraining engine automatically maintains optimizer state, step counter, loss curve, and RNG seeds:

```bash
python scripts/resume_pretraining.py \
    --checkpoint checkpoints/bravien-1.5b-pretrain/checkpoint_step_10000.pt \
    --data-path data/packed_pretrain/bravien_pretrain.pt \
    --output-dir checkpoints/bravien-1.5b-pretrain
```

### D. Supervised Fine-Tuning (SFT) & Multi-turn Chat
After foundation pretraining, train conversation turns using assistant-span loss masking:

```bash
python scripts/train_bravien.py \
    --model-checkpoint checkpoints/bravien-1.5b-pretrain/final.pt \
    --data-path data/processed/sft_chat.jsonl \
    --output-dir checkpoints/bravien-1.5b-sft \
    --epochs 3 \
    --lr 2e-5
```

### E. Direct Preference Optimization (DPO) Alignment
Align model outputs to preference pairs using [`bravien.training.alignment`](file:///c:/Users/LOQ/OneDrive/Documents/Bravien/bravien/training/alignment.py):

```bash
python scripts/align_bravien.py \
    --model-checkpoint checkpoints/bravien-1.5b-sft/final.pt \
    --preference-data data/processed/dpo_preferences.jsonl \
    --output-dir checkpoints/bravien-v4-aligned \
    --beta 0.1 \
    --lr 5e-6
```

---

## 4. Key Environment Variables

| Variable | Description | Example |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | Specifies active GPU IDs | `0,1,2,3` |
| `PYTORCH_CUDA_ALLOC_CONF` | Optimizes memory allocation | `expandable_segments:True` |
| `WANDB_API_KEY` | Weights & Biases telemetry tracking | `<api_key>` |
| `BRAVIEN_CHECKPOINT` | Production checkpoint path | `checkpoints/bravien-v4` |
| `BRAVIEN_MODEL_BACKEND` | Model provider backend | `native` |
