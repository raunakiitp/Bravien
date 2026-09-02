# Bravien Modal Control & Cloud Pretraining Integration

This directory contains the remote cloud execution layer for Bravien using [Modal](https://modal.com). It enables deterministic, cost-safe execution of preflight audits, smoke tests, and bounded foundation pretraining pilots on remote cloud GPUs.

---

## 1. Architecture Overview

```
Local Environment (Antigravity / Developer)
    │
    ├── tools/modal_control.py (CLI interface)
    │
    └── modal/bravien_modal.py (Modal App & Image Definition)
            │
            ├── Base Image: Debian-slim + Python 3.11 + PyTorch + CUDA
            ├── Local Code Mount: /root/Bravien (repo source code, configs, tokenizers)
            └── Persistent Volume: /mnt/bravien (Modal Volume: "bravien-storage")
                    ├── /mnt/bravien/checkpoints/ (atomic step_XXXXXXX checkpoints)
                    ├── /mnt/bravien/reports/     (telemetry summaries)
                    ├── /mnt/bravien/logs/        (runtime logs)
                    └── /mnt/bravien/data/        (packed token datasets)
```

---

## 2. GPU Tiers & Cost Safety Policy

Bravien enforces strict cost safety:
- **Default Tier**: Cheapest available GPU (`T4` at ~$0.59/hr or `L4` at ~$0.80/hr).
- **Prohibited**: Automated selection of high-cost tiers (`A100`, `H100`, or multi-GPU).
- **Transparency**: The chosen GPU tier and estimated hourly cost rate are printed before any remote task executes.
- **Budget Limits**: Training runs are bounded by `--max-steps` or `--max-tokens` to prevent runaway credit consumption.

---

## 3. Available Control Commands

### 1. Workspace Status & Remote Health Check (Zero GPU cost)
Verifies local Modal CLI, authenticated profile, remote container spin-up, and persistent volume mounts:
```bash
python tools/modal_control.py status
```

### 2. Remote Preflight Audit (GPU: T4)
Runs `scripts/preflight_pretraining.py` on a remote GPU container without initiating training:
```bash
python tools/modal_control.py preflight --gpu T4
```

### 3. Remote Smoke Test (10 Steps, bravien-tiny)
Performs a fast 10-step smoke validation on a remote GPU to verify forward pass, loss calculation, backward pass, optimizer step, and volume checkpoint commit:
```bash
python tools/modal_control.py smoke --gpu T4
```

### 4. Bounded Real-Data Pilot (GPU: L4 / A10G)
Runs a bounded foundation pretraining pilot on real packed token data:
```bash
# Step-bounded (50 steps)
python tools/modal_control.py pilot --steps 50 --gpu L4

# Token-bounded (50M tokens)
python tools/modal_control.py pilot --tokens 50000000 --gpu L4
```

### 5. Resume Interrupted Training
Resumes training directly from the latest valid checkpoint on the persistent Modal Volume:
```bash
python tools/modal_control.py resume --gpu L4
```

---

## 4. Modal Storage & Persistent Volume

The persistent volume `bravien-storage` is mounted inside containers at `/mnt/bravien`. All training checkpoints are written atomically (`temp_step_*` $\rightarrow$ `step_*`) and committed to the volume at the end of each session.
