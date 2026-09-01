# Bravien Model Production & Lineage Matrix

---

## 1. Production Model Lineage

```
┌───────────────────────────────────────────────────────────────────────┐
│                      LEGACY BASELINE (ROLLBACK)                       │
├───────────────────────────────────────────────────────────────────────┤
│ • Bravien-v1 (Stage 3): Initial SFT baseline from Qwen2.5-0.5B        │
│ • Bravien-v2 (Stage 6): Targeted training on synthetic QA (24/27)     │
│ • Bravien-v3 (Stage 7-9): Current Production Rollback (494M params)   │
└───────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│                   NATIVE SOVEREIGN ARCHITECTURE                       │
├───────────────────────────────────────────────────────────────────────┤
│ • Bravien-1.5B / Bravien-v4 (Stage 10): Native Architecture (1.509B)  │
│   - Model Type: bravien                                               │
│   - Architecture: BravienForCausalLM                                  │
│   - Layers: 32 | Hidden: 2048 | Heads: 16/4 GQA | Intermediate: 5632  │
│   - Vocab: 32,000 Byte-level BPE                                      │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Hierarchy & Configuration

| Identifier | Class | Parameters | Backend Config | Status |
| :--- | :--- | :--- | :--- | :--- |
| **`bravien-1.5b`** | `BravienForCausalLM` | **1,508,509,696** | `BRAVIEN_MODEL_BACKEND=native` | **Candidate Architecture** |
| **`bravien-v3`** | `Qwen2ForCausalLM` | **494,032,768** | `BRAVIEN_MODEL_BACKEND=qwen_compat` | **Production Rollback Baseline** |
| **`bravien-v2`** | `Qwen2ForCausalLM` | **494,032,768** | `BRAVIEN_CHECKPOINT_PATH=checkpoints/bravien-v2` | **Historical Reference** |
| **`bravien-v1`** | `Qwen2ForCausalLM` | **494,032,768** | `BRAVIEN_CHECKPOINT_PATH=checkpoints/bravien-v1` | **Historical Reference** |

---

## 3. Dynamic Model Info Endpoint (`/api/model/info`)

The server exposes dynamic model discovery so frontend clients accurately report whichever model is actively loaded:

```json
{
  "name": "Bravien-1.5B",
  "architecture": "BravienForCausalLM",
  "parameters": 1508509696,
  "contextLength": 4096,
  "status": "ready",
  "backend": "native"
}
```
