# Bravien 1.5B Training & Data Scaling Roadmap

---

## 1. Explicit Distinction of Milestones

To maintain scientific integrity and engineering rigor, the development of Bravien-1.5B is strictly decoupled into distinct, verifiable phases:

```
[Phase 1: Architecture Foundation]  --> COMPLETE (Current State)
              │
[Phase 2: Large-Scale Corpus Ingestion] (100B - 300B Tokens)
              │
[Phase 3: Foundational Pretraining] (Loss < 2.2 on multi-GPU / offload)
              │
[Phase 4: Continued Domain Enrichment] (Math, Code, STEM, Hindi, Hinglish)
              │
[Phase 5: Supervised Instruction Tuning (SFT)] (Stage 2 51k+ corpus)
              │
[Phase 6: Agent & Tool Alignment (DPO/RLHF)]
              │
[Phase 7: Full 35-Dim Benchmark Verification]
              │
[Phase 8: Production Promotion & Qwen Deprecation]
```

> **CRITICAL DIRECTIVE**: The existing Stage 2 dataset (~51,055 examples / ~9.44M tokens) is an **instruction-tuning corpus**, NOT a pretraining dataset. Pretraining a 1.5B parameter model requires 100 Billion+ tokens.

---

## 2. Phase-by-Phase Roadmap

### Milestone A: Architecture & Foundation (COMPLETE)
- Native model configuration and layers implemented (`BravienForCausalLM`).
- Parameter count programmatically verified at **1,508,509,696 parameters**.
- Native Byte-level BPE tokenizer trained with 32,000 subwords.
- Checkpoint serialization and provider abstraction implemented.

### Milestone B: Pretraining Data Scaling (100B–300B Tokens)
- **General Educational Web**: High-quality filtered subsets from FineWeb / RefinedWeb (50%).
- **Code & Technical Systems**: Permissively licensed Python, TypeScript, Rust, Go, SQL, Bash (20%).
- **Mathematical & STEM Reasoning**: OpenWebMath, Proof-Pile, synthetic reasoning derivations (15%).
- **Hindi & Hinglish Multilingual**: AI4Bharat, Samanantar, curated Hinglish dialogues (10%).
- **Structured Tool Data**: JSON, API schemas, tool calling sequences (5%).

### Milestone C: Pretraining Engine Execution
- Pretraining configuration: [`configs/bravien_1p5b_pretrain.yaml`](file:///c:/Users/LOQ/OneDrive/Documents/Bravien/configs/bravien_1p5b_pretrain.yaml).
- Hardware targeting: Multi-GPU cluster or cloud compute for bulk training; AdamW with Cosine Warmup, 4096 context length, 2M token batch size.

### Milestone D: Continued Pretraining & Specialization (5B–10B Tokens)
- Domain adaptation targeting high-density programming, systems architecture, and bilingual Hindi-English reasoning.

### Milestone E: Supervised Fine-Tuning (SFT)
- Aligning model outputs to assistant personas using the curated Stage 2 corpus (~51,055 examples).
- Masking user prompt loss, training exclusively on assistant response spans.

### Milestone F: Stage 9 Benchmark & Production Evaluation
- Evaluate candidate 1.5B checkpoint against the 35-dimension Stage 9 benchmark.
- Compare side-by-side with `bravien-v3`.
- Promote candidate to primary production ONLY after passing all quality, safety, and latency criteria.
