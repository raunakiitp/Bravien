# Bravien Training Data Strategy (1.5B Native Model)

## 1. Executive Strategy

Training a sovereign **Bravien-1.5B** model from scratch requires a multi-stage curriculum. Under modern scaling laws (e.g. Chinchilla compute-optimal + Llama/Qwen overtraining regime), a 1.5B parameter model requires between **50B and 300B high-quality tokens** for pretraining, followed by targeted continuous domain adaptation and instruction tuning.

```
+----------------------------------------------------------------------------------------------------+
|                                    BRAVIEN DATA CURRICULUM                                         |
+----------------------------------------------------------------------------------------------------+
|  Phase 1: Foundation Pretraining (100B - 300B Tokens)                                              |
|    - General Web & Books (English, Hindi) [45%]                                                    |
|    - Code Repositories & Tech Docs (Python, TS, Rust, Go, SQL) [30%]                               |
|    - STEM, Math, Academic Papers & Reasonings [15%]                                                |
|    - Multilingual & Hinglish Dialogues [10%]                                                       |
+----------------------------------------------------------------------------------------------------+
|  Phase 2: Continued Domain Pretraining (5B - 10B Tokens)                                           |
|    - System Architecture, Local Hardware Optimization, OS Internals                                |
|    - Curated High-Density Synthetic Python & TypeScript Codebases                                  |
|    - Advanced Mathematical Reasoning & Chain-of-Thought Proofs                                     |
+----------------------------------------------------------------------------------------------------+
|  Phase 3: Supervised Fine-Tuning & Alignment (Stage 2 Dataset: 51k+ Examples / ~15M Tokens)         |
|    - Multi-turn Conversational Memory & State Tracking                                             |
|    - Tool Calling, JSON Schemas & Deterministic Execution Formatting                                |
|    - Refusal & Prompt Injection Defenses (Safety Boundaries)                                       |
|    - Code Debugging & Off-by-one Analysis                                                          |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Corpora Categorization

### A. Pretraining Corpus (~100B - 300B Tokens)
- **General Knowledge**: Curated FineWeb / RefinedWeb / OpenWebText extracts filtered for educational value and high perplexity filtering.
- **Multilingual**: High-quality Hindi and Hinglish corpus (Wikipedia, news, bilingual transcripts).
- **License Requirement**: Permissive open licenses (CC-BY, Apache-2.0, MIT, Public Domain).

### B. Continued Pretraining Corpus (~5B - 10B Tokens)
- Specialized technical textbooks, RFCs, POSIX standards, GPU architecture manuals, PyTorch internal documentation, and modern LLM architecture papers.

### C. Instruction Tuning Corpus (Stage 2 Dataset: ~51,055 Examples, ~9.44M Tokens)
- **Role**: SFT fine-tuning of the pretrained foundation model.
- **Components**:
  - `general_qa`: 10,000 examples
  - `coding_tasks`: 10,000 examples (Python, TypeScript, Debugging)
  - `reasoning_math`: 8,000 examples (GSM8K style, step-by-step arithmetic)
  - `multiturn_conversations`: 6,000 multi-turn dialogs
  - `hinglish_interactions`: 5,000 bilingual Hindi/English explanations
  - `instruction_following`: 4,000 format-constrained tasks
  - `tool_use_dialogues`: 3,500 tool invocation and schema formatting pairs
  - `safety_and_refusals`: 2,500 safety refusals and injection resistance examples
  - `anti_hallucination`: 2,055 context-grounded abstentions

### D. Safety & Alignment Corpus
- Strict refusal boundaries against dangerous cyber operations (e.g. DDoS, credential theft), self-harm, and illegal instructions.
- Prompt injection resistance (`[SYSTEM OVERRIDE]` defense, role-play escape resistance).

### E. Coding Corpus
- Syntactically verified Python and TypeScript programs with unit tests.
- Algorithms, data structures, full-stack Next.js web applications, and systems programming.

### F. Hinglish Corpus
- Natural conversational Hinglish ("Python me list aur tuple ka difference samjhao", "Off-by-one error debug kardo").
- Avoids robotic translation; preserves authentic colloquial tech speech.

### G. Agent & Tool Calling Corpus
- Strict structured JSON and XML tool formatting (`<TOOL_CALL>`, `calculator`, `unit_converter`, `database_query`).
- Observation handling, tool error recovery, and sequential multi-step tool execution.

### H. Evaluation Corpus
- Bravien Stage 9 Benchmark (35 comprehensive dimensions).
- MMLU (STEM / Computer Science subsets), HumanEval (Python coding), GSM8K (math reasoning), IFEval (instruction following).

---

## 3. Contamination & Ethics Standards

1. **Zero Secret Ingestion**: All training datasets must pass automated regex filtering for API keys, private tokens, passwords, and PII.
2. **License Compliance**: Every data shard must record license origin and deduplication hash in `bravien/data/pretraining_manifest.py`.
3. **No Private/Copyrighted Violations**: Web data must respect `robots.txt` and permissive licensing terms.
