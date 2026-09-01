# Bravien-v4 Qwen Independence & Model Sovereignty Audit

---

## 1. Zero Dependency Audit Matrix

The audit performed by `scripts/audit_qwen_dependency.py` verifies zero architectural or runtime dependencies on Qwen for Bravien-v4:

| Dimension | Dependency on Qwen | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | **NO (0%)** | `model_type == "bravien"`, Pure PyTorch `BravienForCausalLM` | **PASS** |
| **Model Tokenizer** | **NO (0%)** | Native 32,000 Byte-level BPE (`tokenizers/bravien-native/`) | **PASS** |
| **Inference Runtime** | **NO (0%)** | `BravienNativeProvider` with `BravienKVCache` | **PASS** |
| **Training Pipeline** | **NO (0%)** | Native `BravienPretrainer` & `BravienAligner` | **PASS** |
| **Rollback Compatibility** | **PRESERVED** | `bravien-v3` retained under `checkpoints/bravien-v3` | **INTACT** |

---

## 2. Definitive Qwen Isolation Flag

$$\text{QWEN\_REQUIRED\_FOR\_INFERENCE} = \mathbf{false}$$

Bravien-v4 executes natively without any third-party weight downloading, third-party model classes, or external tokenizer runtimes.
