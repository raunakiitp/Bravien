# Bravien-v4 Production Architecture & Inference Infrastructure

---

## 1. Production Model Hierarchy

The Bravien runtime supports sovereign native inference while maintaining deterministic fallback to legacy checkpoints:

```
[Primary Production / Candidate] --> checkpoints/bravien-v4 (1.508B Native)
                │
                ▼ (Rollback Baseline)
[Production Rollback Model]     --> checkpoints/bravien-v3 (494M Qwen-Compat)
                │
                ▼
[Historical Reference]          --> checkpoints/bravien-v2
                │
                ▼
[Initial SFT Baseline]          --> checkpoints/bravien-v1
```

---

## 2. Model Specifications

- **Model Identifier**: `Bravien-1.5B` / `Bravien-v4`
- **Class**: `BravienForCausalLM`
- **Model Type**: `bravien`
- **Parameter Count**: **1,508,509,696** (~1.508B)
- **Layers**: 32
- **Hidden Size**: 2048
- **Attention Heads**: 16 Query Heads / 4 Key-Value Heads (4:1 GQA)
- **Intermediate Dimension**: 5632 (SwiGLU)
- **Context Length**: 4096 tokens
- **Vocabulary**: 32,000 Byte-level BPE
- **Weight Tying**: Word embeddings tied with output LM head

---

## 3. Inference VRAM Math (RTX 4050 6GB)

- **Model Weights (BF16)**: $1.508\text{B} \times 2\text{ bytes} = \mathbf{2.81\text{ GB}}$
- **4096-token KV Cache (4:1 GQA)**: $\mathbf{0.27\text{ GB}}$
- **CUDA Buffers & Activations**: $\mathbf{\sim 0.50\text{ GB}}$
- **Peak Active VRAM**: $\mathbf{\sim 3.58\text{ GB}}$
- **Headroom on 6GB RTX 4050**: $\mathbf{2.42\text{ GB}}$ (**40.3% free headroom**).

---

## 4. Integration with Agent, Tools, Memory & Intercept

1. **Deterministic Intercept Layer (Stage 9)**: Pre-screens arithmetic, unit conversion, and prompt injection before model invocation.
2. **Autonomous Agent (Stage 8)**: Executes bounded multi-step tool sequences (Calculator, Memory, RAG).
3. **Local Memory System**: Relevance-ranked, local-first storage with automatic secret rejection.
4. **Streaming Protocol (Stage 5)**: Server-Sent Events (SSE) streaming with active lifecycle state frames.
