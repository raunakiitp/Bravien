# Bravien Stage 8: Repository & Architecture Audit

## 1. Executive Summary

This audit establishes the baseline architectural state of the Bravien system prior to the Stage 8 Autonomous Local Intelligence and Capability upgrade.

---

## 2. Current Subsystems & Production Pipelines

### 2.1 Model Loading & Inference Engine
- **Inference Runtime**: FastAPI server in `bravien/inference/server.py` and `HFInferenceEngine` in `bravien/inference/hf_engine.py`.
- **Current Production Checkpoint**: `checkpoints/bravien-v3` (`494,032,768` parameters, `torch.bfloat16`, loaded on NVIDIA GeForce RTX 4050 Laptop GPU with ~3.0GB VRAM).
- **Rollback Hierarchy**: `checkpoints/bravien-v3` → `checkpoints/bravien-v2` → `checkpoints/bravien-v1` → `Qwen/Qwen2.5-0.5B-Instruct`.

### 2.2 Orchestration & Routing
- **Agent Orchestration**: `src/lib/ai/agent-orchestrator.ts` and `src/lib/ai/orchestrator.ts`.
- **Deterministic Gating**: `src/lib/ai/inference-gate.ts` (intercepts ping, persona identity, direct calculations, time queries before model inference).
- **Response Caching**: `src/lib/ai/inference-cache.ts` (tenant-isolated LRU semantic cache).

### 2.3 Verification & Anti-Hallucination
- **Verification Layer**: `src/lib/ai/verification.ts` (arithmetic calculation verification, document grounding ratio check, code bracket and indentation verification).

### 2.4 Tool Execution
- **Existing Tools**: `src/lib/ai/tools/` (`calculator.ts`, `time.ts`, `documents.ts`, `memories.ts`, `web.ts`, `loop.ts`, `registry.ts`).

### 2.5 Training & Data Pipeline
- **Dataset Pipeline**: `bravien/data/pipeline.py`, `bravien/data/curation.py`, `bravien/data/synthetic.py` (43,994 validated examples across train/val/test splits with SHA-256 manifest verification).
- **Trainer**: `bravien/training/hf_trainer.py` supporting `bfloat16`, gradient accumulation, learning rate warmup, cosine decay, and assistant-only loss masking.

---

## 3. Identified Limitations to Address in Stage 8

1. **Tool Registry Fragmentation**: Tools are defined partially in TypeScript and ad-hoc in Python without a unified schema contract.
2. **Intent Classification**: Routing currently relies on keyword regex heuristics; needs a formal hybrid deterministic + heuristic + model-assisted `IntentResult` classification layer.
3. **Multi-Step Execution**: Agent loop needs bounded step-by-step planning, observation ingestion, and self-correction verification (max 8 steps).
4. **Structured Memory**: Persistent memory lacks importance scoring, explicit type separation (preference vs project vs session), and strict secret exclusion policies.
5. **UI Activity Transparency**: UI needs a non-intrusive, expandable "Agent Activity" summary accordion that shows completed steps without exposing internal system prompts or raw chains of thought.
6. **Model Independence**: Model loading logic needs complete abstraction (`ModelProvider` / `BravienLocalProvider`) so future model checkpoints remain drop-in replacements.

---

## 4. File Modification Boundaries

### Files to Create / Upgrade:
- `bravien/agent/intent.py` & `src/lib/ai/intent.ts` (Intent classification)
- `bravien/tools/` (`__init__.py`, `registry.py`, `schemas.py`, `executor.py`, `permissions.py`, `errors.py`) (Unified Tool Registry)
- `bravien/agent/tool_selector.py` (Tool selection intelligence)
- `bravien/agent/task_executor.py` (Multi-step execution)
- `bravien/agent/verification.py` & `src/lib/ai/verification.ts` (Self-correction loop)
- `bravien/memory/` (`memory_store.py`, `memory_types.py`, `memory_retriever.py`, `memory_policy.py`) (Structured memory)
- `bravien/agent/context.py` (Context token budgeting)
- `bravien/model/provider.py` & `src/lib/ai/providers/base.ts` (Model provider abstraction)
- `src/components/chat/message.tsx` (Expandable Agent Activity UI & smart response actions)
- `scripts/stage8_benchmark.py` & `scripts/test_stage8_agent.ts` (Stage 8 benchmark & regression suite)
- `docs/architecture/stage8-autonomous-intelligence.md`, `stage8-operations.md`, `stage8-observability.md`

### Files to PRESERVE without destructive modification:
- `checkpoints/bravien-v3/` (Active production checkpoint)
- `checkpoints/bravien-v2/` & `checkpoints/bravien-v1/` (Rollback baselines)
- `src/lib/ai/inference-gate.ts` & `src/lib/ai/inference-cache.ts`
- `scripts/test_stage4.ts`, `test_stage5_ux.ts`, `test_stage6_intelligence.ts`, `test_stage7_intelligence.ts`, `test_phase15.ts`
