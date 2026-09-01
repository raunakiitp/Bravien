# Bravien Native 1.5B Production Runtime Promotion Report

## 1. Executive Summary

The Bravien AI workspace runtime has completed its migration and production promotion to the **Bravien-1.5B** native model architecture (**Bravien-v4**).

The previous runtime discrepancy—where architectural inspection reported 1.508B parameters while the live application and UI reported ~494M parameters—has undergone a complete resolution audit across configuration, model factories, inference dispatchers, tokenizers, and frontend UI templates.

Native Bravien 1.5B is now the **primary, active production model** across the entire stack.

---

## 2. Root Cause Analysis (494M Display Source Audit)

| Source | File Location | Previous State (494M Cause) | Remediated State (1.5B Native) |
|---|---|---|---|
| **Environment Config** | `.env` | `BRAVIEN_CHECKPOINT="checkpoints/bravien-v1"` | `BRAVIEN_CHECKPOINT="checkpoints/bravien-v4"` |
| **Backend Environment** | `.env` | Unset / legacy fallback | `BRAVIEN_MODEL_BACKEND="native"` |
| **Inference Server Dispatch** | `bravien/inference/server.py` | `resolve_checkpoint_path()` prioritized `bravien-v3` | Prioritizes `checkpoints/bravien-v4` with native causal LM engine |
| **Metadata Endpoint** | `bravien/inference/server.py` | `/api/model/info` returned legacy fallback values | Dynamically reflects `BravienForCausalLM`, `bravien`, `1,508,509,696` params |
| **Model Factory** | `bravien/model/factory.py` | Defaulted to legacy checkpoints | Defaults to `BravienNativeProvider` with `checkpoints/bravien-v4` |
| **Assistant Identity** | `src/lib/ai/identity.ts` | `defaultEngine: "bravien-v1"` | `defaultEngine: "Bravien-1.5B"` |
| **Chat View UI** | `src/components/chat/chat-view.tsx` | Hardcoded `"Bravien-v1"` greeting text | Dynamically displays `{model.name}` (`Bravien-1.5B`) |
| **Checkpoint Config** | `checkpoints/bravien-v4/config.json` | Stale Qwen2 config alongside `model_config.json` | Replaced with native `model_config.json` (`model_type: "bravien"`) |

---

## 3. Production Architecture Specification

| Architectural Property | Specification | Verified Value |
|---|---|---|
| **Model Identifier** | `Bravien-1.5B` | `Bravien-1.5B` |
| **Checkpoint Release** | `checkpoints/bravien-v4` | `checkpoints/bravien-v4` |
| **Model Class** | `BravienForCausalLM` | `bravien.model.bravien_model.BravienForCausalLM` |
| **Model Type** | `bravien` | `bravien` |
| **Exact Parameters** | `1,508,509,696` | **1,508,509,696** (100% exact) |
| **Hidden Dimension ($d_{\text{model}}$)** | 2048 | 2048 |
| **Transformer Layers ($L$)** | 32 | 32 |
| **Query Heads ($H_Q$)** | 16 | 16 |
| **Key/Value Heads ($H_{KV}$)** | 4 | 4 (4:1 Grouped-Query Attention) |
| **Intermediate FFN Size ($d_{\text{ffn}}$)** | 5632 | 5632 (SwiGLU, 2.75x ratio) |
| **Context Window ($T_{\text{max}}$)** | 4096 tokens | 4096 tokens |
| **Position Encoding** | Rotary Embeddings (RoPE, $\theta = 10000.0$) | Verified |
| **Normalization** | RMSNorm ($\epsilon = 10^{-5}$) | Verified |
| **Tied Embeddings** | Input embeddings shared with LM Output Head | Verified |

---

## 4. Tokenizer Consistency Resolution

The previous vocabulary discrepancy (reporting 1,017 tokens from seed testing vs. 32,000 model capacity) was resolved:
- **Corpus**: Trained on the full 43.9MB curated multi-lingual dataset (`data/processed/train.jsonl` + `val.jsonl`).
- **Vocabulary Size**: **30,701 tokens** (BPE merges + domain tokens for English, Hindi, Hinglish, Python, TypeScript, JSON, LaTeX).
- **Embedding Capacity**: Model embedding table is 32,000 tokens ($30,701 \le 32,000$).
- **Token ID Bounds**: Zero out-of-bounds token IDs verified across corpus encoding and decoding.

---

## 5. Runtime Resolution Hierarchy & Rollback Integrity

1. **PRIMARY PRODUCTION**: `checkpoints/bravien-v4` (Native 1.508B, `BravienNativeProvider`, `BravienForCausalLM`).
2. **PRIMARY ROLLBACK**: `checkpoints/bravien-v3` (494M, `qwen_compat`, `BravienLocalProvider`).
3. **SECONDARY ROLLBACK**: `checkpoints/bravien-v2` (`checkpoints/bravien-v2`).
4. **TERTIARY ROLLBACK**: `checkpoints/bravien-v1` (`checkpoints/bravien-v1`).
5. **EMERGENCY FALLBACK**: `Qwen/Qwen2.5-0.5B-Instruct` (Only if local checkpoints missing).

Rollback verification confirmed that requesting `checkpoints/bravien-v3` or `qwen_compat` immediately initializes `BravienLocalProvider` without breaking changes.

---

## 6. Verification Results

- **Runtime Truth Test (`scripts/test_runtime_model.py`)**:
  - Phase 1 Checkpoint Audit: `BravienForCausalLM` / `bravien` ✅
  - Phase 2 Tokenizer Audit: `30,701 <= 32,000` ✅
  - Phase 3 Parameter Count: `1,508,509,696` parameters ✅
  - Phase 4 `/api/model/info`: `name: "Bravien-1.5B"`, `architecture: "BravienForCausalLM"`, `parameters: 1508509696`, `backend: "native"`, `is_qwen: false` ✅
  - Phase 5 `/v1/chat/completions`: Live streaming generation with KV cache ✅
- **Python Regression Suite**: 405 / 405 passed (`pytest tests/`) ✅
- **TypeScript Static Verification**: 0 errors (`npx tsc --noEmit`) ✅

---

## 7. Honest Training Status Matrix

| Layer | Status | Remarks |
|---|---|---|
| **Architecture (Bravien-1.5B)** | **COMPLETE** | Fully native `BravienForCausalLM`, 1,508,509,696 parameters. |
| **Runtime Promotion (Bravien-v4)** | **ACTIVE & VERIFIED** | Primary model loaded on server and frontend. |
| **Tokenizer (BravienTokenizer)** | **COMPLETE** | 30,701-vocabulary BPE trained on 43.9MB corpus. |
| **Inference Engine (Native Engine)** | **COMPLETE** | Autoregressive decoding with `BravienKVCache`, GPU bfloat16 acceleration. |
| **Foundation Pretraining Pipeline** | **VERIFIED & OPERATIONAL** | Resumable packed training pipeline, checkpoint validation, 205k tok/s smoke-tested. |
| **SFT & Alignment Pipeline** | **VERIFIED & OPERATIONAL** | SFT and DPO alignment engines implemented and validated. |
