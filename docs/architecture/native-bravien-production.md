# Bravien Native Production Architecture Specification

---

## 1. Overview & Executive Summary

Bravien has completed its architectural evolution from an experimental fine-tuned research baseline to a **fully native, sovereign ~1.5 Billion parameter causal language model ecosystem**.

```
                           ┌───────────────────────────────────────────────┐
                           │               BRAVIEN PLATFORM                │
                           └───────────────────────┬───────────────────────┘
                                                   │
                                     ┌─────────────▼─────────────┐
                                     │    Bravien Native 1.5B    │
                                     │  (Causal LM Architecture) │
                                     └─────────────┬─────────────┘
                                                   │
                   ┌───────────────────────────────┼───────────────────────────────┐
                   │                               │                               │
       ┌───────────▼───────────┐       ┌───────────▼───────────┐       ┌───────────▼───────────┐
       │   Bravien Tokenizer   │       │   Agent Orchestrator  │       │    Context Manager    │
       │    (Byte-Level BPE)   │       │   (Routing & Intercept│       │   (Budget & Memory)   │
       └───────────────────────┘       └───────────┬───────────┘       └───────────────────────┘
                                                   │
                                       ┌───────────┴───────────┐
                                       │                       │
                           ┌───────────▼───────────┐ ┌─────────▼───────────┐
                           │   Deterministic Tools │ │   Grounding & RAG   │
                           │  (Calc, Units, Time)  │ │ (Vector Retrieval)  │
                           └───────────────────────┘ └─────────────────────┘
                                                   │
                                       ┌───────────▼───────────┐
                                       │   Verification & UX   │
                                       │   (Safety, SSE Stream)│
                                       └───────────────────────┘
```

---

## 2. Why Bravien No Longer Depends on Qwen

| Dimension | Previous Baseline (Bravien-v1/v2/v3) | Native Bravien 1.5B (Current Architecture) |
| :--- | :--- | :--- |
| **Model Weights** | Initialized from `Qwen/Qwen2.5-0.5B-Instruct` | **100% Bravien-owned native initialization** |
| **Tokenizer** | Qwen BPE Tokenizer (`qwen2.tiktoken`) | **Native Byte-level BPE Tokenizer (`BravienTokenizer`)** |
| **Architecture Classes** | `transformers.models.qwen2.Qwen2ForCausalLM` | **Pure native `BravienForCausalLM` in `bravien/model/`** |
| **Runtime Inference** | HuggingFace AutoModel pipeline | **Native token-by-token streaming engine with `BravienKVCache`** |
| **External Dependencies** | Dependent on HuggingFace downloads at runtime | **Zero network calls; 100% local self-contained weights** |
| **Rollback Strategy** | N/A | **`bravien-v3` retained as explicit manual fallback** |

---

## 3. Native Model Architecture (~1.509B Parameters)

- **Layers**: 32 Transformer Decoder Blocks
- **Hidden Dimension**: 2048
- **Attention Heads**: 16 Query heads, 4 Key-Value heads (**4:1 Grouped-Query Attention**)
- **Head Dimension**: 128
- **Feed-Forward Network**: SwiGLU Gated MLP with intermediate dimension 5632 (2.75x)
- **Context Window**: 4096 tokens
- **Positional Embeddings**: Rotary Position Embeddings (RoPE, $\theta = 10,000.0$)
- **Normalization**: Pre-RMSNorm ($\epsilon = 10^{-5}$)
- **Weight Tying**: Word embeddings tied with output linear layer

### Parameter Breakdown
```
Token Embeddings:   65,536,000
Self-Attention:    335,544,320
SwiGLU MLP:      1,107,296,256
RMSNorms:              133,120
Output LM Head:              0 (Tied to input embeddings)
--------------------------------------------------------
Total Parameters: 1,508,509,696 (~1.509 Billion)
```

---

## 4. Production Inference Engine (`bravien/model/native_provider.py`)

- **Streaming Token Generation**: High-throughput token-by-token generator yielding delta strings over SSE.
- **Dynamic KV Caching**: Layer-by-layer `BravienKVCache` eliminating redundant prefix computation during autoregression.
- **Sampling Controls**: Full support for `temperature`, `top_p`, `top_k`, `repetition_penalty`, and `seed`.
- **Cancellation & Safety**: Graceful abort handling when the user triggers the client-side Stop button.

---

## 5. Agent Orchestrator & Deterministic Intercept Layer

To ensure mathematical precision, safety, and instant response times:
1. **Deterministic Intercept Pre-Screening**:
   - Direct arithmetic expressions ($45 \times 12 + 60 \rightarrow 600$) evaluated via safe AST in $<0.2\text{ms}$.
   - Unit and temperature conversions ($100^\circ\text{C} \rightarrow 212^\circ\text{F}$) computed with exact formulas.
   - Prompt injection defense (`[SYSTEM OVERRIDE]` / jailbreaks) stopped deterministically.
   - Ambiguous queries prompted for user clarification before inference.
2. **Model Reasoning & Synthesis**:
   - Complex reasoning, coding synthesis, Hinglish explanation, and conversation flow handled by `BravienForCausalLM`.

---

## 6. Memory & Context Budgeting

- **Memory System**: Contextually recalls user preferences, session history, and workspace configurations while enforcing secret redaction.
- **RAG Grounding**: Grounded document retrieval attaches source citations and halts absent-fact hallucinations.

---

## 7. Configuration & Rollback Operational Protocol

### Default Native Production Configuration
```env
BRAVIEN_MODEL_BACKEND=native
BRAVIEN_NATIVE_CHECKPOINT=checkpoints/bravien-native-1.5b
BRAVIEN_ENABLE_QWEN_COMPAT=false
```

### Explicit Manual Rollback Procedure
If emergency fallback to the legacy `bravien-v3` baseline is required:
```env
BRAVIEN_MODEL_BACKEND=qwen_compat
BRAVIEN_CHECKPOINT_PATH=checkpoints/bravien-v3
BRAVIEN_ENABLE_QWEN_COMPAT=true
```
