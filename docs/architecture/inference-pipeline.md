# Bravien Native Local Model Inference Pipeline

## Overview

Bravien runs a local-first AI assistant pipeline where an instruction-tuned 0.5B-scale language model (`Qwen2.5-0.5B-Instruct` or native checkpoint) acts as the generative reasoning brain, backed by deterministic tool execution, persistent AgentState, RAG retrieval, and web research.

## Request Flow

```
User Prompt (Chat UI / useChat)
      ↓
POST /api/chat (SSE Stream with AbortSignal)
      ↓
Unified Agent Orchestrator (runUnifiedAgentTurn)
      ↓
Intent Classification & Execution Mode Selection
  ├── DIRECT                → Fast conversational response
  ├── TOOL                  → Deterministic math / time / lookup
  ├── RESEARCH              → Verified web search & citation extraction
  ├── PLANNED               → Bounded multi-step plan execution
  ├── TASK                  → Autonomous task runner with checkpoints
  └── WAITING_CONFIRMATION  → High-risk action confirmation gate
      ↓
Evidence Gathering (RAG, Web, Memories, AgentState)
      ↓
Canonical Context Builder & Token Budgeter (formatModelPrompt)
  Priority Degradation:
  1. Drop oldest conversation turns (preserve latest user query)
  2. Reduce web evidence
  3. Reduce RAG document context
  4. Reduce memories
  5. Compact agent state
  6. Preserve system rules & identity
      ↓
ModelRuntime Singleton (modelRuntime.streamGenerate)
      ↓ HTTP (loopback:8000)
FastAPI Local Inference Server (bravien.inference.server)
      ↓
InferenceEngine / HFInferenceEngine (CUDA RTX 4050 / CPU fallback)
      ↓ SSE Frames
Real-time Events (agent_event, citation, content_delta, message_complete)
      ↓
Database Persistence (Prisma / PostgreSQL) & Chat UX
```

## Model Runtime Lifecycle & Concurrency

- **State Transitions**: `IDLE` → `LOADING` → `READY` → `BUSY` → `FAILED`
- **Concurrency Control**: Semaphore-bounded generation slots (`maxConcurrency = 1` for local GPU memory safety).
- **Promise Coalescing**: Concurrent `loadModel()` invocations share the same active initialization Promise.
- **Warmup & Health**: Non-blocking `GET /api/ai/health` and explicit kernel warmup `POST /api/ai/warmup`.

## Generation Profiles

| Profile | Temperature | Top-P | Repetition Penalty | Max Tokens | Use Case |
|---|---|---|---|---|---|
| `FAST` | 0.1 | 0.85 | 1.05 | 512 | Math, time, tool synthesis, factual chat |
| `BALANCED` | 0.4 | 0.90 | 1.10 | 1024 | General conversation, reasoning, research |
| `CREATIVE` | 0.7 | 0.95 | 1.15 | 1536 | Creative writing, brainstorming |
| `CODE` | 0.1 | 0.85 | 1.05 | 2048 | Code generation, debugging, refactoring |

## Deterministic Guardrails vs. Model Responsibilities

To maximize quality on a 0.5B model:
- **Deterministic Systems**: Arithmetic, time, database search, document retrieval, web scraping, SSRF validation, authorization, action risk classification.
- **Model Engine**: Natural language synthesis, explanation, summarization, evidence reasoning, coding responses.

## Error Handling & Sanitization

Structured error codes protect the user experience and ensure security:
- `MODEL_UNAVAILABLE`: Runtime offline or checkpoint unloaded.
- `MODEL_BUSY`: Maximum concurrent generations active.
- `MODEL_TIMEOUT`: Generation duration limit exceeded.
- `MODEL_CANCELLED`: Request cleanly aborted by client via `AbortSignal`.
- `MODEL_GENERATION_FAILED`: Unrecoverable inference backend error (sanitized without leaking paths or secrets).
