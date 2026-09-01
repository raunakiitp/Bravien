# Bravien Stage 8: Autonomous Local Intelligence, Tool Use & Capability Architecture

## 1. System Architecture Overview

Bravien Stage 8 establishes a complete local-first autonomous intelligence architecture. It unifies intent classification, a real tool registry, safe execution lifecycles, bounded multi-step task planning, risk-based self-verification with a 2-attempt self-correction loop, structured memory with secret rejection, and context token budgeting.

```
USER REQUEST
    │
    ▼
DETERMINISTIC GATE (ping, identity, direct arithmetic)
    │ (miss)
    ▼
INTENT CLASSIFICATION (bravien/agent/intent.py & src/lib/ai/intent.ts)
    │ (13 intent categories)
    ▼
TOOL SELECTOR (hybrid deterministic + heuristic selection)
    │
    ├── Direct Answer ──────────► CONTEXT BUILDER ──► MODEL INFERENCE ──► SELF-VERIFY ──► RESPONSE
    │
    └── Tool Required ──────────► TOOL REGISTRY (schemas, permissions, timeouts)
                                      │
                                      ▼
                                  SAFE EXECUTION ENGINE (errors masked)
                                      │
                                      ▼
                                  OBSERVATION & CONTEXT SYNTHESIS
                                      │
                                      ▼
                                  SELF-CORRECTION LOOP (max 2 retries)
                                      │
                                      ▼
                                  FINAL STREAMED SSE RESPONSE
```

---

## 2. Core Capabilities & Subsystems

### 2.1 Unified Intent Classification (`bravien/agent/intent.py`, `src/lib/ai/intent.ts`)
Classifies user turns into 13 structured categories:
- `conversation`, `factual_question`, `reasoning`, `mathematics`, `coding`, `document_question`, `memory`, `planning`, `tool_required`, `multi_step_task`, `clarification_required`, `unsafe_request`, `unknown`.
- Exposes `IntentResult`: `intent`, `confidence`, `requires_tool`, `requires_memory`, `requires_rag`, `requires_reasoning`, `requires_confirmation`, `safety_level`.

### 2.2 Unified Tool Registry (`bravien/tools/`)
- 10 Built-in Safe Tools: `calculator`, `unit_converter`, `datetime`, `text_transform`, `json_parser`, `document_retrieval`, `memory_read`, `memory_write`, `code_validation`, `structured_planning`.
- Strict Permissions & Error Masking: Command injection blocking, confirmation requirement for high-risk actions, internal stack trace and secret redaction.

### 2.3 Bounded Multi-Step Task Executor (`bravien/agent/task_executor.py`)
- Executes multi-step tasks in a bounded `PLAN -> STEP -> OBSERVE -> VERIFY -> FINALIZE` cycle.
- Default `max_steps = 8` preventing infinite agent execution loops.

### 2.4 Structured Memory & Secret Filtering (`bravien/memory/`)
- Memory Types: `preference`, `project`, `fact`, `session`, `conversation`.
- Security Boundary: Automatically rejects passwords, auth tokens, API keys, private keys, and credit cards from memory persistence.
- Selective Retrieval: Relevance scoring and top-K injection avoiding context bloat.

### 2.5 Risk-Based Self-Verification & Correction (`bravien/agent/verification.py`, `src/lib/ai/verification.ts`)
- Evaluates arithmetic calculations, code bracket balance and syntax, document grounding ratios, and honest abstentions.
- Triggers targeted self-correction before emitting final answer.
