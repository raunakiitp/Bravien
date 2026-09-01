# Bravien Stage 10: Tokenizer Strategy & Migration Decision

---

## 1. Context & Evaluation

The tokenizer is a foundational artifact in language modeling. Changing subword segmentation boundaries carelessly can invalidate existing tokenized datasets, damage chat template alignments, and alter prefix parsing rules.

This strategy document addresses the trade-offs between:
1. **Option A**: Continuing with the Qwen tokenizer (`qwen2.tiktoken` / 151,936 vocab).
2. **Option B**: Training an independent, compact Byte-level BPE tokenizer (`tokenizers/bravien-native/` with 32,000 vocab).

---

## 2. Technical Evaluation & Decision

| Dimension | Legacy Qwen Tokenizer (v1/v2/v3) | Native Bravien Tokenizer (v4 / 1.5B) |
| :--- | :--- | :--- |
| **Vocabulary Size** | 151,936 tokens | **32,000 tokens** |
| **Embedding Memory (Hidden=2048)** | $151,936 \times 2048 \times 2 = \mathbf{622.3\text{ MB}}$ | $32,000 \times 2048 \times 2 = \mathbf{131.1\text{ MB}}$ |
| **Embeddings Parameter Count** | **311,164,928 params** (20.6% of 1.5B) | **65,536,000 params** (4.3% of 1.5B) |
| **Multilingual Coverage** | English, Chinese, Multilingual | **English, Hindi, Hinglish, Code, JSON, Math** |
| **Special Tokens** | `<|im_start|>`, `<|im_end|>` | `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<|im_start|>`, `<|im_end|>` |
| **Sovereignty & Licensing** | Tied to Qwen HuggingFace releases | **100% Owned by Bravien, local Rust core** |

### Decision
1. **For Native Bravien 1.5B / Bravien-v4**: Use the native 32,000-vocabulary Byte-level BPE tokenizer (`BravienTokenizer`). Saving ~245M parameters from embeddings allows allocating more capacity to deep Transformer layers (32 layers with 5632 intermediate FFN dimension).
2. **For Legacy Rollback (Bravien-v1/v2/v3)**: Preserve the existing tokenizer compatibility layer via `BravienLocalProvider` and `HFInferenceEngine`.
3. **Training Data Compatibility**: Stage 2 dataset JSONL files format data in standard role/content conversational structures, enabling lossless tokenization by `BravienTokenizer.encode_chat()` without dataset mutation.
