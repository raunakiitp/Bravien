# Bravien Stage 6: Intelligence & Model Quality Upgrade

## Executive Summary

Stage 6 implements an empirical, deterministic intelligence evaluation benchmark across 12 critical capabilities for the local Bravien AI assistant, identifies concrete failure modes in `bravien-v1`, scales targeted dataset quality, and trains/validates `bravien-v2` with strict non-regression safeguards.

---

## 1. 12-Category Deterministic Intelligence Benchmark

The Bravien Intelligence Benchmark (`bravien.evaluation.benchmarks.stage6_benchmark`) defines 27 rigorous test items evaluated across 12 distinct dimensions:

| # | Dimension | Description | Target Standard |
|---|---|---|---|
| 1 | **Identity & Persona** | Local-first assistant identity, hardware locality, zero third-party claims | Clear self-identification as Bravien, local device execution |
| 2 | **Normal Conversation** | Polite greetings, gratitude acknowledgment, open-ended conversational flow | Natural, helpful conversational tone |
| 3 | **Multi-Turn Memory** | Cross-turn variable tracking, user preference recall | Retaining entities and preferences across turns |
| 4 | **Factual QA** | Encyclopedic and scientific precision | Exact factual recall (e.g. capitals, astronomy, biology) |
| 5 | **Reasoning & Math** | Multi-step arithmetic, percentage discounts, rate problems | Correct final numeric calculation and step-by-step logic |
| 6 | **Coding** | Algorithms, complexity analysis, language syntax (Python, TS, SQL) | Idiomatic code, correct Big-O analysis |
| 7 | **Hinglish Assistance** | Natural Hindi-English mixed technical explanations | Accurate bilingual code and conceptual explanations |
| 8 | **Instruction Following** | Negative constraints, exact bullet formatting, strict JSON output | Strict adherence to user formatting directives |
| 9 | **Uncertainty & Abstention** | Honest refusal on unknowable future predictions, missing data | Explicitly stating lack of access/unpredictability |
| 10 | **Safety & Refusal** | Refusal of malicious cyber attacks (DDoS, phishing, key theft) | Safe ethical refusal with defensive guidance |
| 11 | **Prompt Injection** | Resistance to jailbreaks, `[SYSTEM OVERRIDE]`, instruction hijacking | Rejection of adversarial overrides without leaking instructions |
| 12 | **Anti-Hallucination** | Document grounding, refusal to fabricate unmentioned facts | Grounding answers strictly within provided context |

---

## 2. Baseline Evaluation (`checkpoints/bravien-v1`)

- **Evaluation Date**: August 31, 2026
- **Model Checkpoint**: `checkpoints/bravien-v1` (Base: `Qwen/Qwen2.5-0.5B-Instruct`)
- **Overall Score**: **20/27 (74.1%)** in 37.00s

### Empirical Diagnosis:
1. **Prompt Injection (0/2 - 0.0%)**:
   - The model printed hijacked attacker tokens when commanded with adversarial override directives (`PWNED_BY_ATTACKER_99`).
2. **Reasoning & Math (1/3 - 33.3%)**:
   - Calculated wrong final price on percentage discounts ($120 - 25% = $90).
   - Inaccurate arithmetic on multi-step equations ($\sqrt{144} + 15 = 27$).
3. **Multi-turn Memory (1/2 - 50.0%)**:
   - Failed to recall user's stated favorite programming language across multiple turns.
4. **Identity & Persona (1/2 - 50.0%)**:
   - Claimed computations ran on a "remote server" rather than the user's local device.
5. **Uncertainty & Abstention (1/2 - 50.0%)**:
   - Did not explicitly declare future asset/market valuations as unknowable/unpredictable.

---

## 3. Targeted Training Data Scaling

To address the diagnosed failure modes without degrading existing capabilities:
- **Curation Seeds**: Added targeted examples in `bravien/data/curation.py` covering adversarial override resistance, percentage discounts, rate word problems, multi-turn preference retention, local offline execution, and future uncertainty abstention.
- **Procedural Scaling**: Expanded `bravien/data/synthetic.py` with randomized generators for injection resistance, arithmetic reasoning, and multi-turn variable tracking.
- **Dataset Pipeline**: Generated 43,993 verified examples across train (37,399), val (3,265), and test (3,329) splits with SHA-256 manifest verification.

---

## 4. Candidate Model Training (`checkpoints/bravien-v2`)

- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Precision**: `torch.bfloat16`
- **Batch Size**: 2 x 8 (effective batch size: 16)
- **Max Sequence Length**: 512 tokens
- **Optimizer**: AdamW ($\text{LR} = 2 \times 10^{-5}$, cosine decay with linear warmup)
- **VRAM Utilization**: ~3.0 GB on NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)

---

## 5. Verification & Rollback Protocol

- `checkpoints/bravien-v1` is preserved as the rollback baseline.
- `bravien-v2` is promoted to production default only after passing:
  1. Identical 27-item benchmark with higher overall score.
  2. Zero regression across all 12 capability dimensions.
  3. TypeScript end-to-end integration test suite (`scripts/test_stage6_intelligence.ts`).
