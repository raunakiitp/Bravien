# Bravien 1.5B Native Architecture Foundation

---

## 1. Why the 494M Model Is Being Replaced

The early versions of Bravien (Bravien-v1, v2, v3) were fine-tuned from `Qwen/Qwen2.5-0.5B-Instruct` (~494M parameters). While this served as an effective fast research baseline to build and test:
- Local streaming and SSE wire protocols
- Intent classification and tool orchestration
- Deterministic intercept gates and safety boundaries
- RAG grounding and memory hierarchies

A ~494M parameter foundation model inherently suffers from:
1. **Limited Knowledge Capacity**: 494M parameters cannot represent deep world knowledge, multi-domain factual depth, and rich reasoning without excessive reliance on external context.
2. **Third-Party Architectural Dependency**: The model relied on Qwen's specific tokenizer vocabulary, tokenization artifacts, and model class implementations.
3. **Sovereignty & Licensing**: A genuinely sovereign AI assistant must own its weights, initialization, training curriculum, and subword tokenizer.

Bravien-1.5B (~1.509B parameters) provides a **3.05x capacity expansion**, designed specifically for local private inference on modern consumer GPUs such as the **NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)**.

---

## 2. Architecture Comparison: Bravien-v3 vs Bravien-1.5B

| Metric / Dimension | Bravien-v3 (Current Baseline) | Bravien-1.5B (Native Architecture) |
| :--- | :--- | :--- |
| **Origin** | Qwen2.5-0.5B-Instruct fine-tune | **100% Bravien Native Initialization** |
| **Total Parameters** | 494,032,768 (~494M) | **1,508,509,696 (~1.509B)** |
| **Layers** | 24 | **32** |
| **Hidden Dimension** | 896 | **2048** |
| **Attention Heads (Q / KV)** | 14 / 2 (7:1 GQA) | **16 / 4 (4:1 GQA)** |
| **Head Dimension** | 64 | **128** |
| **FFN Intermediate Dimension** | 4864 | **5632 (SwiGLU)** |
| **Context Length** | 2048 tokens | **4096 tokens** |
| **Vocabulary Size** | 151,936 | **32,000 (Lossless Byte-level BPE)** |
| **Embeddings** | Untied (136.1M params) | **Tied Word Embeddings (65.5M params)** |
| **Weight Precision** | BF16 (0.98 GB) | **BF16 (2.81 GB)** |

---

## 3. Detailed Architectural Design

### A. Attention with 4:1 Grouped-Query Attention (GQA)
- **16 Query heads** and **4 Key-Value heads** share KV representations across 4 query heads.
- **KV Cache Memory Math**:
  $$\text{KV Cache per token per layer} = 2 \times (\text{num\_kv\_heads} \times \text{head\_dim}) \times 2\text{ bytes} = 2 \times (4 \times 128) \times 2 = 2,048\text{ bytes} = 2\text{ KB}$$
  $$\text{Full 4096-token KV cache for 32 layers} = 32 \times 4096 \times 2048\text{ bytes} \approx 268.4\text{ MB}$$
- Compared to Multi-Head Attention (16 KV heads $\rightarrow$ 1.07 GB cache), 4:1 GQA delivers a **75% reduction in KV cache VRAM footprint**.

### B. SwiGLU Gated Feed-Forward Network
- Uses the Gated Linear Unit with SiLU activation:
  $$\text{FFN}(x) = \text{down\_proj}(\text{SiLU}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x))$$
- Intermediate dimension is set to $5632$ ($\approx \frac{8}{3} \times d_{\text{model}}$), delivering optimal capacity per FLOP.

### C. Normalization & Positional Embeddings
- **Pre-RMSNorm**: Root Mean Square Layer Normalization with $\epsilon = 10^{-5}$ applied before attention and MLP sub-layers.
- **Rotary Position Embeddings (RoPE)**: Applied to query and key states with base frequency $\theta = 10,000.0$.

---

## 4. Parameter Calculation Breakdown

Analytical and PyTorch tensor verification confirms exact parameter counts:

```
Embedding Parameters:        32,000 * 2048                =    65,536,000  (4.3%)
Attention Parameters:        32 * [2048*(16+4+4+16)*128]  =   335,544,320 (22.2%)
SwiGLU MLP Parameters:       32 * [3 * (2048 * 5632)]     = 1,107,296,256 (73.4%)
RMSNorm Parameters:          (2 * 2048 * 32) + 2048       =       133,120 (<0.1%)
LM Head Parameters:          Tied to Word Embeddings      =             0  (0.0%)
---------------------------------------------------------------------------------
TOTAL PARAMETERS:                                           1,508,509,696 (1.509B)
```

Target Window: **1.45B – 1.60B** $\rightarrow$ **STATUS: PASS**

---

## 5. Tokenizer Strategy & Multilingual Coverage

- **Technology**: Byte-level Byte-Pair Encoding (BPE) with HuggingFace `tokenizers` Rust core.
- **Vocabulary Size**: 32,000 subwords.
- **Reserved Special Tokens**:
  - `PAD`: ID `0` (`<PAD>`)
  - `UNK`: ID `1` (`<UNK>`)
  - `BOS`: ID `2` (`<BOS>`)
  - `EOS`: ID `3` (`<EOS>`)
  - `SYSTEM_START`: `<|im_start|>system`
  - `USER_START`: `<|im_start|>user`
  - `ASSISTANT_START`: `<|im_start|>assistant`
  - `TURN_END`: `<|im_end|>`
- **Tested Languages & Modalities**:
  - English: 100% lossless roundtrip
  - Hindi (Devanagari script): 100% lossless roundtrip
  - Hinglish (Latin script Hindi-English): 100% lossless roundtrip
  - Python & TypeScript: 100% lossless roundtrip
  - JSON & Mathematical LaTeX: 100% lossless roundtrip

---

## 6. Hardware Feasibility on NVIDIA RTX 4050 (6GB VRAM)

### Inference Feasibility
- **1.5B Weights in BF16**: **2.81 GB**
- **4096-token KV Cache (4:1 GQA)**: **0.27 GB**
- **Activation Buffers & CUDA Overhead**: **~0.50 GB**
- **Total Peak Inference VRAM**: **~3.58 GB**
- **VRAM Headroom on 6GB RTX 4050**: **~2.42 GB (40.3% free headroom)**.

### Training Realism & Strategy
- Full FP32 AdamW training of a 1.5B model requires:
  - Weights (BF16): 2.81 GB
  - Gradients (BF16): 2.81 GB
  - AdamW Optimizer States (FP32 moments $m$ and $v$): $2 \times 4\text{ bytes} \times 1.5\text{B} = \mathbf{12.0\text{ GB}}$
- Therefore, full pretraining on 6GB consumer hardware is staged via:
  1. Multi-GPU / external compute for foundational pretraining
  2. 8-bit AdamW / CPU optimizer offloading
  3. LoRA / QLoRA parameter-efficient fine-tuning on local RTX 4050
  4. Local development verification using `bravien-tiny` and `bravien-small` presets.

---

## 7. Migration & Rollback Strategy

The system preserves the full rollback hierarchy:
1. **Primary Production**: `bravien-v3` (494M Qwen-compatible fine-tune) remains active until the 1.5B pretraining and SFT curriculum are fully trained and evaluated.
2. **Candidate Backend**: `bravien-1.5b` accessible via `BRAVIEN_MODEL_BACKEND=native`.
3. **Rollback Baseline**: `bravien-v3` $\rightarrow$ `bravien-v2` $\rightarrow$ `bravien-v1` $\rightarrow$ `Qwen2.5-0.5B-Instruct`.

---

## 8. Exact Verification Commands

```bash
# 1. Verify exact 1.5B parameter breakdown
python scripts/count_bravien_parameters.py --preset bravien-1.5b --instantiate-check

# 2. Test native tokenizer lossless roundtrip
python scripts/test_bravien_tokenizer.py

# 3. Run full 25-point 1.5B test suite
python -m pytest tests/test_bravien_1_5b.py -v

# 4. Run entire repository test suite
python -m pytest tests/ -q
npx tsc --noEmit
```
