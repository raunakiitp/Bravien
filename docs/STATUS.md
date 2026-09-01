# Bravien Project Status & Reproducibility Guide

## 1. What is Bravien?

Bravien is a privacy-first, local-first conversational AI workspace and native language model system designed to run on personal hardware with complete data sovereignty and zero external API dependencies.

---

## 2. Model Architecture & Parameters

| Property | Specification | Verified Value |
|---|---|---|
| **Model Name** | `Bravien-1.5B` | `Bravien-1.5B` |
| **Model Class** | `BravienForCausalLM` | `bravien.model.bravien_model.BravienForCausalLM` |
| **Model Type** | `bravien` | `bravien` |
| **Exact Parameters** | `1,508,509,696` | **1,508,509,696** (1.5085B) |
| **Hidden Size ($d_{\text{model}}$)** | 2048 | 2048 |
| **Layers ($L$)** | 32 | 32 |
| **Query Heads ($H_Q$)** | 16 | 16 |
| **KV Heads ($H_{KV}$)** | 4 | 4 (4:1 Grouped-Query Attention) |
| **Intermediate Size ($d_{\text{ffn}}$)** | 5632 | 5632 (SwiGLU, 2.75x ratio) |
| **Max Context ($T_{\text{max}}$)** | 4096 tokens | 4096 tokens |
| **Position Encoding** | Rotary Embeddings (RoPE, $\theta = 10000.0$) | RoPE |
| **Normalization** | RMSNorm ($\epsilon = 10^{-5}$) | RMSNorm |
| **Tied Embeddings** | Shared Input Embedding & Output Head | Verified |

---

## 3. Runtime & Checkpoints

- **Active Production Checkpoint**: `checkpoints/bravien-v4`
- **Active Backend**: `native` ([`bravien.inference.native_engine.NativeInferenceEngine`](file:///c:/Users/LOQ/OneDrive/Documents/Bravien/bravien/inference/native_engine.py))
- **Provider**: [`bravien.model.provider.BravienNativeProvider`](file:///c:/Users/LOQ/OneDrive/Documents/Bravien/bravien/model/provider.py)
- **Rollback Hierarchy**:
  1. Primary: `checkpoints/bravien-v4` (Native ~1.508B)
  2. Rollback: `checkpoints/bravien-v3` (494M Qwen-derived baseline, `BravienLocalProvider`)
  3. Secondary: `checkpoints/bravien-v2`
  4. Tertiary: `checkpoints/bravien-v1`
  5. Emergency Fallback: `Qwen/Qwen2.5-0.5B-Instruct`

---

## 4. Tokenizer Status

- **Type**: Byte-Level Byte Pair Encoding (BPE)
- **Tokenizer Class**: [`bravien.tokenizer.tokenizer.BravienTokenizer`](file:///c:/Users/LOQ/OneDrive/Documents/Bravien/bravien/tokenizer/tokenizer.py)
- **Vocabulary Size**: **30,701 tokens** (Trained on full 43.9MB multilingual & code corpus)
- **Embedding Table Capacity**: **32,000 tokens** ($30,701 \le 32,000$)
- **Special Tokens**: `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`, `<USER>`, `</USER>`, `<ASSISTANT>`, `</ASSISTANT>`, `<SYSTEM>`, `</SYSTEM>`, `<TOOL_CALL>`, `</TOOL_CALL>`, `<TOOL_RESULT>`, `</TOOL_RESULT>`

---

## 5. Implementation Status Matrix

| Component | Status | Details |
|---|---|---|
| **Native Architecture** | **COMPLETE** | 1,508,509,696 parameters, fully independent `BravienForCausalLM`. |
| **Native Runtime** | **ACTIVE & PROMOTED** | Primary runtime in server, model factory, and Next.js frontend. |
| **Native Tokenizer** | **COMPLETE** | 30,701 vocabulary with byte fallback and complete special token support. |
| **Training Engine** | **COMPLETE & TESTED** | Resumable packed pretraining, cosine LR, BF16 autocast, gradient accumulation. |
| **SFT Engine** | **COMPLETE & TESTED** | Assistant-span loss masking and multi-turn chat template formatting. |
| **Alignment Engine** | **COMPLETE & TESTED** | DPO preference loss optimization engine (`bravien/training/alignment.py`). |
| **Agent / Tools / Memory** | **COMPLETE** | Autonomous task loop, RAG vector index, deterministic intercept layer. |
| **Full Foundation Pretraining** | **READY FOR CLOUD** | Code & pipeline verified via smoke pretraining (205k tok/s); large-scale token pretraining to run on cloud GPUs. |

---

## 6. Test & Validation Results

- **Unit Tests**: **405 / 405 passed** in 35.38s (`pytest tests/`)
- **Runtime Truth Test**: **100% Passed** (`scripts/test_runtime_model.py`)
- **TypeScript Static Analysis**: **0 errors** (`npx tsc --noEmit`)
- **Model Parameters Check**: Verified exactly 1,508,509,696 parameters.

---

## 7. How to Run Locally

### A. Environment Setup
```bash
# Python dependencies
pip install torch transformers fastapi uvicorn pydantic pytest

# Node.js dependencies
npm install
```

### B. Validate Architecture & Runtime Truth
```bash
# Verify model parameter count, tokenizer bounds, and live endpoints
python scripts/test_runtime_model.py
```

### C. Run Native Inference Server
```bash
# Serves checkpoints/bravien-v4 on http://127.0.0.1:8000
python scripts/serve.py --checkpoint checkpoints/bravien-v4
```

### D. Run Full Regression Suite
```bash
# Run all 405 unit tests
python -m pytest tests/

# Run TypeScript type check
npx tsc --noEmit
```

### E. Run Training Smoke Test
```bash
# Pack pretraining tokens
python scripts/pack_pretraining_data.py --input data/processed/train.jsonl --output data/packed_pretrain/smoke.pt

# Run pretraining smoke test
python scripts/resume_pretraining.py --data-path data/packed_pretrain/smoke.pt --max-steps 10 --batch-size 2
```
