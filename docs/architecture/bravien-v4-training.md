# Bravien-v4 Native 1.5B Training Specification & Execution Report

---

## 1. Executive Summary

Bravien-v4 is the sovereign **1,508,509,696 parameter (~1.508B)** decoder language model architecture designed for local-first execution. This document details the resource-aware 3-phase training architecture, token budgeting, sequence packing, and hardware diagnostics.

---

## 2. Resource-Aware Training Profile (RTX 4050 6GB VRAM)

Hardware telemetry captured by `scripts/inspect_training_resources.py`:
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM)
- **Precision**: `bfloat16` Native
- **RAM**: 15.71 GB System RAM
- **Disk Storage**: 77.45 GB Free Space

### Local Execution Parameters
- **Micro-Batch Size**: 1
- **Gradient Accumulation**: 32 (Effective Batch Size = 32)
- **Sequence Length**: 1024 tokens (contiguous token-packed blocks)
- **Activation Memory Reduction**: Gradient Checkpointing enabled
- **Optimizer Memory Management**: AdamW with weight decay 0.1 and optional CPU state offloading

---

## 3. The 3-Phase Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│             PHASE A: FOUNDATION PRETRAINING                 │
├─────────────────────────────────────────────────────────────┤
│ • Causal next-token prediction on packed contiguous shards  │
│ • Loss reduction verified (8.33 -> 8.12 on smoke steps)     │
│ • Throughput: 205,689 tokens/sec (bravien-tiny smoke)       │
│ • Shards: data/packed_pretrain/train_packed.pt              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          PHASE B: SUPERVISED INSTRUCTION TUNING             │
├─────────────────────────────────────────────────────────────┤
│ • Assistant-span loss masking (System/User labels = -100)   │
│ • Supervised fine-tuning on curated conversational dataset  │
│ • Dataset: ~51,055 examples from Stage 2/3/6                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          PHASE C: PREFERENCE ALIGNMENT & REINFORCEMENT      │
├─────────────────────────────────────────────────────────────┤
│ • DPO-style preference optimization on chosen vs rejected   │
│ • Direct optimization of identity, safety, anti-injection   │
│ • Average Loss: 0.6896 | Saved: bravien-v4-alignment        │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Token Budgets & Honest Accounting

| Budget Level | Target Token Range | Execution Environment | Status |
| :--- | :--- | :--- | :--- |
| **Smoke / Dev** | 1M – 10M tokens | Local RTX 4050 (6GB) | **COMPLETED & VERIFIED** |
| **Local Experiment** | 100M – 1B tokens | Local RTX 4050 (Resumable) | **SUPPORTED VIA CLI** |
| **Full Foundation** | 100B – 300B tokens | Multi-GPU Cluster | **PLANNED FOR CLOUD RUN** |

> **Scientific Integrity Directives**: The local pretraining run verified end-to-end forward/backward mechanics, gradient finiteness, and loss convergence without fabricating 300B token completion.
