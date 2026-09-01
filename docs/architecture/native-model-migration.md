# Native Model Migration Plan: Qwen-Derived to Native Bravien 1.5B

## 1. Architectural Comparison Matrix

| Dimension | Current Production (`bravien-v3`) | Target Native (`bravien-1.5b`) |
|---|---|---|
| **Base Foundation** | `Qwen/Qwen2.5-0.5B-Instruct` (Alibaba) | **Bravien Sovereign Architecture** (Clean native PyTorch) |
| **Parameter Count** | ~494 Million (0.494B) | **1,508.51 Million (1.509B)** |
| **Model Weight Provenance** | Fine-tuned from external base weights | **Pretrained from scratch on Bravien curated corpus** |
| **Tokenizer** | Qwen BPE (152,064 tokens) | **Bravien Native Byte-Level BPE (32,000 tokens)** |
| **Attention Architecture** | 14 Q Heads, 2 KV Heads (7:1 GQA) | **16 Q Heads, 4 KV Heads (4:1 GQA, head_dim=128)** |
| **Layer Depth** | 24 Transformer Blocks | **32 Transformer Blocks** |
| **Hidden Dimension** | 896 | **2048** |
| **SwiGLU Intermediate** | 4864 (~5.4x) | **5632 (~2.75x)** |
| **Context Length** | 2048 tokens standard | **4096 tokens standard** |
| **Checkpoint Serialization** | HuggingFace Safetensors + `config.json` | **Bravien Native Checkpoint + SHA-256 Manifest** |
| **Runtime Dependencies** | `transformers`, `torch` | **`torch` (Zero `transformers` runtime dependency)** |
| **Backend Flag** | `BRAVIEN_MODEL_BACKEND=qwen_compat` | `BRAVIEN_MODEL_BACKEND=native` |

---

## 2. Phased Migration Timeline

```
[Phase 1: Foundation Completed] (Current State)
  ├── Native Model Architecture: bravien/model/bravien_*.py
  ├── Native Tokenizer: bravien/tokenizer/ (Byte-level BPE, 100% roundtrip)
  ├── Native Checkpointing: bravien/model/checkpoint.py (SHA-256 verified)
  ├── Pretraining Engine: bravien/training/pretrain.py (BF16, AdamW, Cosine, 49k+ tok/s)
  └── Multi-Backend Model Provider: bravien/model/provider.py

[Phase 2: Full Tokenizer & Pretraining Corpus Curation]
  ├── Build 100B+ token pretraining manifest (manifests/pretraining_manifest.json)
  ├── Train final 32,000-token production vocabulary (tokenizers/bravien-native-prod)
  └── Validate token compression and Devanagari/Code fertility

[Phase 3: Large-Scale Distributed Pretraining (1.5B Parameters)]
  ├── Launch pretraining run on target multi-GPU compute cluster
  ├── Save rolling checkpoints to checkpoints/bravien-native-1.5b-base/
  └── Evaluate convergence curve & validation perplexity (<15.0 target)

[Phase 4: SFT Instruction Tuning & Agent Alignment]
  ├── Train SFT checkpoint on Stage 2 Dataset (51,055 examples + Stage 9 seeds)
  ├── Align deterministic tool calling, multi-turn state, and prompt injection defense
  └── Validate on Stage 9 Benchmark (Target: >= 33/35 native generation score)

[Phase 5: Seamless Production Cutover]
  ├── Set BRAVIEN_MODEL_BACKEND=native in production deployment configs
  ├── Keep qwen_compat as emergency fallback
  └── Sunset legacy Qwen dependencies
```

---

## 3. Operational CLI Commands Reference

### A. Run Parameter Analysis
```bash
python scripts/count_bravien_parameters.py --preset bravien-1.5b
```

### B. Train / Retrain Native Tokenizer
```bash
python scripts/train_bravien_tokenizer.py --output-dir tokenizers/bravien-native --vocab-size 32000
```

### C. Test Tokenizer Lossless Compression
```bash
python scripts/test_bravien_tokenizer.py --tokenizer-dir tokenizers/bravien-native
```

### D. Run Native Smoke Pretraining (Fast Sanity Check)
```bash
python scripts/pretrain_bravien.py --tiny-smoke
```

### E. Run Full 1.5B Pretraining (When Cluster Hardware is Available)
```bash
python scripts/pretrain_bravien.py --preset bravien-1.5b --micro-batch-size 4 --gradient-accumulation-steps 16 --lr 3e-4 --output-dir checkpoints/bravien-native-1.5b
```

### F. Run Native Model Benchmark Evaluation
```bash
python scripts/evaluate_native_bravien.py --checkpoint checkpoints/native_smoke_test/step_0000020
```
