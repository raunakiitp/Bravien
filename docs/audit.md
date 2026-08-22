# Bravien — Phase 0 Audit

Date: 2026-08-22
Auditor: automated repository inspection (pre-implementation)

## Method

Full working-tree inspection, `git log --all --name-only` over every file ever
committed, toolchain execution (`tsc --noEmit`), and host hardware probing.

## Correction to the stated premise

The build brief states the repository "also contains existing Bravien
development/training work" and asks that a "previous Bravien training pipeline
or v0.x implementation" be built upon rather than discarded.

**No such work exists.** The repository contains zero Python files, zero
notebooks, zero checkpoints, zero tokenizer artifacts, and zero ML code — not in
the working tree, and not at any point in git history (verified across all refs).
The only non-TypeScript config is `docker-compose.yml` (Postgres).

Everything on the machine-learning side is therefore greenfield. Nothing is
being discarded, because there is nothing to discard. This is reported here
rather than silently glossed, per the honesty requirement (§71, §72).

## Current architecture

A single-commit `create-next-app` scaffold, plus one large uncommitted
foundation layer (now committed as `feat: scaffold data, auth, and AI
foundation`).

| Layer | State |
| --- | --- |
| Next.js 16.3.2 / React 19.2.8 / TS 5 | Present, compiles clean |
| Prisma schema (Postgres) | Complete, 14 models — no migrations generated |
| Auth.js v5 (JWT + Prisma adapter) | Complete: credentials + optional Google/GitHub |
| `lib/ai/*` orchestration | Present, but points at hosted APIs (see below) |
| `lib/tools`, `lib/files`, `lib/memory`, `lib/search`, `lib/security` | Written, unreferenced by any route |
| `components/ui/*` (24 primitives on `@base-ui/react`) | Present, unused |
| App routes | **Only** the Auth.js catch-all exists |
| Pages / screens | None — `app/page.tsx` is the create-next-app default |
| Tests | None. `npm test` runs `vitest`, which is not installed |

## What is already functional

- `tsc --noEmit` passes with zero errors.
- Prisma schema is coherent and models the full product surface: users,
  conversations, branching messages (self-relation), attachments, memories,
  tool executions, usage records, feedback, API keys, subscriptions, flags.
- Auth.js wiring is correct and edge-safe: `auth.config.ts` holds a
  Prisma-free config for middleware, full providers live in `auth.ts`.
- `src/types/index.ts` defines a genuinely good streaming abstraction:
  `AIProvider.streamText() -> AsyncIterable<AIStreamChunk>`, with a
  discriminated-union chunk type covering content deltas, tool start/result,
  citations, completion and error. **This is reusable verbatim** — a local
  Bravien runtime provider can implement it and map cleanly onto SSE.
- `lib/ai/context.ts` (history trimming) and `lib/ai/prompts.ts` (system prompt
  assembly) are model-agnostic and worth keeping.

## Architectural conflict — must be resolved, not preserved

`src/lib/ai/models.ts` defines a four-entry model registry:

| Bravien-branded id | Actually resolves to |
| --- | --- |
| `bravien-fast` | `gpt-4o-mini` |
| `bravien-balanced` | `gpt-4o` |
| `bravien-reasoning` | `o3-mini` |
| `bravien-vision` | `gpt-4o` |

`src/lib/ai/providers/index.ts` resolves all of these through
`OpenAICompatibleProvider` using `OPENAI_API_KEY`.

This is precisely the forbidden topology of §76 — `Bravien -> another LLM ->
pretend it is Bravien`. It is the single largest thing this build must correct.
The registry must instead report the *real* architecture of the loaded local
checkpoint, read from the Bravien runtime, and the default provider must be the
local inference server.

The existing `openai-compatible.ts` adapter is retained but demoted to an
isolated, non-default, explicitly-labelled experimental path (permitted by §2).

## Incomplete / missing functionality

- No inference engine, model, tokenizer, training loop, data pipeline, or
  evaluation — the entire ML stack.
- No local API server.
- No product UI: no chat screen, composer, sidebar, message list, settings,
  or auth screens.
- No API routes for conversations, messages, files, or memories — although
  `middleware.ts` and `auth.config.ts` already declare guards for
  `/api/conversations`, `/api/chat`, `/api/files`, `/api/memories`.
- No Prisma migrations (`prisma/migrations/` absent).
- No test infrastructure at all.
- `docs/` did not exist before this file.

## Technical debt

- `package.json` declares `"test": "vitest run"` with no `vitest` dependency.
- `.gitignore` contains the `uploads` ignore block twice.
- `app/layout.tsx` metadata still reads "Create Next App".
- `lib/features.ts` defaults `vision: true` and `file_uploads: true` for
  capabilities that do not exist yet — these should default off until the
  runtime genuinely implements them (§44, §72).
- `.env.example` documents `FEATURE_MEMORY`/`FEATURE_TOOLS`/`FEATURE_SHARING`/
  `FEATURE_API_KEYS`, but `lib/features.ts` reads a different key set
  (`FEATURE_WEB_SEARCH`, `FEATURE_VISION`, ...). The two disagree.
- `lib/ai/models.ts` reads `BRAVIEN_MODEL_BALANCED`/`_VISION`, while
  `.env.example` documents `BRAVIEN_MODEL_DEFAULT`/`_SMART`/`_ANTHROPIC`.
  Also drifted.

## Host hardware (probed)

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6141 MiB VRAM, Ada (sm_89) |
| Driver | 610.62, CUDA UMD 13.3 |
| CPU | Intel i5-13450HX |
| RAM | 16.9 GB |
| Free disk | 53 GB |
| Python | 3.14.5 |
| PyTorch | not installed; `2.9.0+cu129` confirmed available for cp314/win |

Ada supports BF16, so mixed-precision training is available. A ~40M parameter
model trains comfortably within 6 GB; this sets the realistic target size.

Already installed and reusable: `tokenizers` 0.23.1 (Rust BPE trainer — real
tokenizer training, no external service), `fastapi` 0.139, `uvicorn` 0.51,
`numpy` 2.5.1, `pydantic` 2.13, `pytest` 9.1.1.

## Recommended migration

1. Add the Python `bravien/` package alongside the existing web app; do not
   disturb the Next.js tree.
2. Build the ML stack bottom-up: config -> model -> tokenizer -> data ->
   training -> inference -> API.
3. Replace `lib/ai/models.ts` and `lib/ai/providers/index.ts` so the default
   path targets the local Bravien runtime; keep the hosted adapter isolated
   and off by default.
4. Build the product UI against the runtime's streaming endpoint.
5. Gate every capability claim on a real runtime implementation.
