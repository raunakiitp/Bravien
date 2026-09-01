# Bravien Stage 10: Native Architecture Audit

---

## 1. Executive Summary & Objective

The objective of Stage 10 is to eliminate the ~494M parameter architectural ceiling inherited from `Qwen/Qwen2.5-0.5B-Instruct` and establish a **sovereign, native ~1.509B parameter Transformer architecture** (`BravienForCausalLM`).

This audit analyzes every subsystem across the repository to map dependencies, delineate legacy HuggingFace/Qwen paths from native Bravien paths, and ensure zero-regression backward compatibility with `bravien-v3` rollback capabilities.

---

## 2. Current Architecture & Parameter Breakdown

### Legacy Baseline (Bravien-v1 / v2 / v3)
- **Base Architecture**: `Qwen2ForCausalLM` (HuggingFace Transformers)
- **Parameter Count**: **494,032,768 parameters (~494M)**
- **Structure**: 24 layers, hidden dimension 896, 14 Query heads, 2 KV heads (7:1 GQA), intermediate dimension 4864, vocabulary 151,936.
- **Weights Precision**: BF16 (~0.98 GB weights).
- **Location**: `checkpoints/bravien-v3/` (production baseline).

### Sovereign Native Architecture (Bravien-1.5B / Bravien-v4)
- **Model Class**: `bravien.model.bravien_model.BravienForCausalLM`
- **Config**: `bravien.model.bravien_config.BravienConfig` (`model_type = "bravien"`, `architecture = "BravienForCausalLM"`)
- **Parameter Count**: **1,508,509,696 parameters (~1.509B)**
- **Structure**: 32 layers, hidden dimension 2048, 16 Query heads, 4 KV heads (4:1 GQA), intermediate dimension 5632 (SwiGLU), vocabulary 32,000 (Lossless Byte-level BPE), tied embeddings.
- **Weights Precision**: BF16 (~2.81 GB weights).
- **Location**: `checkpoints/bravien-native-1.5b/` (candidate architecture).

---

## 3. Dependency Mapping & Code Classification

### A. Core Model & Tokenizer Subsystems
| Component | File Path | Current Status | Stage 10 Action |
| :--- | :--- | :--- | :--- |
| **Model Configuration** | `bravien/model/bravien_config.py` | Native `BravienConfig` (`bravien-1.5b`) | ✅ Standardize & expose alias in `bravien/model/config.py` |
| **Model Layers** | `bravien/model/bravien_layers.py`, `attention.py`, `mlp.py`, `norm.py` | Pure PyTorch (RoPE, SwiGLU, RMSNorm, GQA) | ✅ Expose `modeling_bravien.py` exports |
| **Parameter Counter** | `scripts/count_bravien_parameters.py` | Analytical & PyTorch meta-device counter | ✅ Package into `bravien/model/parameter_count.py` & `scripts/model_info.py` |
| **Model Factory** | `bravien/model/provider.py` | Provider abstraction | ✅ Create `bravien/model/factory.py` for dynamic resolution |
| **Tokenizer** | `bravien/tokenizer/tokenizer.py` | Native Byte-level BPE (32k vocab) | ✅ Document migration & lossless bilingual support |

### B. Training & Inference Subsystems
| Subsystem | File Path | Dependency | Stage 10 Status |
| :--- | :--- | :--- | :--- |
| **Native Pretrainer** | `bravien/training/pretrain.py` | Pure PyTorch + AMP | ✅ 100% independent of Qwen classes |
| **Legacy SFT Trainer** | `bravien/training/hf_trainer.py` | HuggingFace Trainer | 🛡️ Preserved for legacy v1/v2/v3 rollback |
| **Inference Server** | `bravien/inference/server.py` | Dual loading (InferenceEngine / NativeProvider) | ✅ Supports explicit `BRAVIEN_MODEL` and dynamic metadata |
| **Provider Layer** | `bravien/model/provider.py` | `BravienNativeProvider` + `BravienLocalProvider` | ✅ Defaults to `native`; preserves opt-in rollback |

### C. Agent, Tools, Memory & Intercept Subsystems
- **Deterministic Intercept**: `bravien/agent/intercept.py` (Math, Unit conversion, Prompt injection, Ambiguity) $\rightarrow$ 100% architecture-agnostic.
- **Agent Orchestrator & Tools**: `src/lib/ai/` (Calculator, Unit converter, Memory, RAG) $\rightarrow$ 100% architecture-agnostic.
- **Stage 4–9 Test Suites**: Validates deterministic gates, safety refusals, streaming lifecycle, and rollback baseline.

---

## 4. Components Requiring Migration vs Safe to Preserve

```
┌──────────────────────────────────────────────┐
│           PRESERVED (NO CHANGES)             │
├──────────────────────────────────────────────┤
│ • checkpoints/bravien-v3/ (Rollback baseline) │
│ • Stage 2 Training Dataset (51k examples)    │
│ • Deterministic Intercept Layer (Stage 9)    │
│ • Memory, RAG & Tool Execution Engines       │
│ • SSE Wire Protocol & Chat UI Components     │
└──────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          MIGRATED / NEW IN STAGE 10          │
├──────────────────────────────────────────────┤
│ • bravien/model/config.py (Alias & Factory)  │
│ • bravien/model/modeling_bravien.py          │
│ • bravien/model/parameter_count.py           │
│ • bravien/model/factory.py                   │
│ • scripts/model_info.py & model_memory.py    │
│ • /api/model/info dynamic metadata route     │
│ • tests/test_bravien_1_5b_architecture.py    │
│ • scripts/test_stage10_architecture.py       │
└──────────────────────────────────────────────┘
```

---

## 5. Architectural Feasibility & Hardware Constraints (RTX 4050 6GB)

- **Weight Footprint (BF16)**: $1.508\text{B} \times 2\text{ bytes} = \mathbf{2.81\text{ GB}}$
- **KV Cache Footprint (4096 tokens, 4:1 GQA)**: $\mathbf{0.27\text{ GB}}$
- **CUDA & Activation Overhead**: $\mathbf{\sim 0.50\text{ GB}}$
- **Total Peak Active Inference VRAM**: $\mathbf{\sim 3.58\text{ GB}}$ ($\sim 2.42\text{ GB}$ headroom on 6GB RTX 4050).
- **Training Feasibility**:
  - Full AdamW states on 1.5B parameters require $12.0\text{ GB}$ VRAM.
  - Large-scale pretraining is staged for multi-GPU/cloud compute or 8-bit AdamW / CPU offload.
  - Local validation runs on `bravien-tiny` (0.81M) and `bravien-small` (39.59M).

---

## 6. Migration Plan & Safety Principles

1. **Explicit Identification**: The 1.5B architecture reports `model_type="bravien"`, `architecture="BravienForCausalLM"`.
2. **Deterministic Rollback**: If `bravien-1.5b` is not selected or candidate testing is active, production safely serves `checkpoints/bravien-v3`.
3. **Zero Fabrications**: Explicit distinction between "1.5B Architecture Created" vs "1.5B Fully Pretrained".
