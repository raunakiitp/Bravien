# Bravien Training Data Foundation Pipeline (Stage 1)

This document outlines the architecture, data schemas, filtering invariants, deduplication strategies, and reproduction steps for the Bravien local model training dataset pipeline.

---

## 1. Pipeline Overview

The Bravien dataset pipeline ingests heterogeneous raw datasets and public-domain corpora, cleans and normalizes text, executes multi-stage safety and quality filtering, performs MinHash and exact deduplication, and deterministically splits the output into train, validation, and test subsets.

```
+------------------+     +------------------------+     +--------------------+
| Raw JSONL / Text | --> | Ingestion & Normalize | --> | Quality & Secrets  |
| Curated Seeds    |     | (bravien.data.clean)   |     | (bravien.data.filt)|
+------------------+     +------------------------+     +--------------------+
                                                                  |
                                                                  v
+------------------+     +------------------------+     +--------------------+
|  Manifest & Hash | <-- | Deterministic Split    | <-- | Exact & MinHash    |
| (data/manifests) |     | (85% / 7.5% / 7.5%)    |     | Deduplication      |
+------------------+     +------------------------+     +--------------------+
```

---

## 2. Canonical Bravien Training Schema

Every example emitted by the pipeline follows the canonical schema:

```json
{
  "id": "e3b0c44298fc1c14",
  "source": "alpaca_data",
  "category": "coding",
  "messages": [
    {
      "role": "system",
      "content": "You are Bravien, a secure, local-first AI assistant..."
    },
    {
      "role": "user",
      "content": "Write a Python function to compute Fibonacci numbers."
    },
    {
      "role": "assistant",
      "content": "Here is an efficient recursive memoized implementation..."
    }
  ],
  "quality_score": 0.95,
  "language": "en",
  "metadata": {
    "imported_from": "alpaca_data"
  }
}
```

### Schema Rules:
- **Roles**: Must be `system`, `user`, `assistant`, or `tool`.
- **Invariants**: Must contain at least one `user` message and must end with an `assistant` turn (the supervised training target).
- **Quality Score**: Normalized float from $0.0$ to $1.0$.

---

## 3. Data Categories

1. **General Knowledge (`general_knowledge`)**: Science, history, philosophy, geography, and general facts.
2. **Instruction Following (`instruction_following`)**: Constraint satisfaction, formatting (JSON/tables), summarization.
3. **Dialogue (`dialogue`)**: Contextual multi-turn conversational interactions.
4. **Reasoning (`reasoning`)**: Step-by-step logic, mathematical deduction, rate calculations.
5. **Coding (`coding`)**: Python, TypeScript, algorithms, debugging, type safety.
6. **Safety & Refusal (`safety_refusal`)**: Ethical refusals for destructive operations (`rm -rf /`, credential exfiltration) with constructive alternatives.
7. **Bravien Persona (`bravien_assistant`)**: Local-first assistant, honest uncertainty, offline execution awareness.
8. **Tool Use (`tool_use`)**: Calculator execution, RAG/document retrieval queries, web search synthesis.
9. **Hinglish (`hinglish`)**: Natural bilingual English/Hindi conversational assistance.

---

## 4. Quality & Safety Filtering

- **Secret & Credential Detection**: Rejection of private keys (`BEGIN PRIVATE KEY`), GitHub PATs (`ghp_`), AWS keys (`AKIA`), and real auth tokens.
- **Repetition & Entropy Filter**: Rejection of degenerate loops using 4-gram and 5-gram frequency ratios.
- **PII Redaction**: Automatic replacement of credit card numbers, SSNs, phone numbers, and emails with anonymized tokens.
- **Formatting Validation**: Balanced code blocks, proper sentence casing, and minimum character thresholds.

---

## 5. Deduplication & Split Determinism

- **Exact Deduplication**: SHA-256 hash of normalized lowercase whitespace-collapsed conversation text.
- **Near Deduplication**: 128-permutation MinHash with Jaccard similarity threshold ($0.85$).
- **Zero Contamination**: Deterministic salted hash splitting prevents prompt or conversation overlap between `train`, `val`, and `test` splits.

---

## 6. How to Reproduce

### Run Dataset Preparation:
```bash
python scripts/prepare_dataset.py --output-dir data/processed --manifest-dir data/manifests --sample-dir data/samples
```

### Validate Dataset:
```bash
python scripts/validate_dataset.py --dataset-dir data/processed --manifest data/manifests/dataset_manifest.json
```

### Inspect Dataset Statistics:
```bash
python scripts/dataset_stats.py --dataset-dir data/processed
```
