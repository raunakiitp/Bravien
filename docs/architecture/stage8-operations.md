# Bravien Stage 8: Operations, Testing & Rollback Guide

## 1. Starting the System

### 1.1 Local Inference Server
```powershell
python scripts/serve.py --checkpoint checkpoints/bravien-v3 --host 127.0.0.1 --port 8000
```

### 1.2 Web Application Dev Server
```powershell
npm run dev
```

---

## 2. Running Test Suites

### 2.1 Complete Automated Regression Matrix
```powershell
# Python unit and integration suite (315+ tests)
python -m pytest

# TypeScript static typecheck
npm run typecheck

# Stage 8 Autonomous Agent Suite
npx tsx scripts/test_stage8_agent.ts

# Stage 7 Intelligence Suite
npx tsx scripts/test_stage7_intelligence.ts

# Stage 6 Intelligence Suite
npx tsx scripts/test_stage6_intelligence.ts

# Stage 5 UX & Streaming Suite
npx tsx scripts/test_stage5_ux.ts

# Stage 4 Production Suite
npx tsx scripts/test_stage4.ts

# Phase 15 Agent Suite
npx tsx scripts/test_phase15.ts
```

### 2.2 Running the 24-Dimension Stage 8 Benchmark
```powershell
python scripts/stage8_benchmark.py --model checkpoints/bravien-v3 --output data/reports/stage8_benchmark_v3.json
```

---

## 3. Rollback & Model Switching Protocol

Bravien preserves an explicit rollback hierarchy:
- Active production checkpoint: `checkpoints/bravien-v3`
- Verified baseline rollback: `checkpoints/bravien-v2`
- Initial fine-tuned rollback: `checkpoints/bravien-v1`
- Foundation fallback: `Qwen/Qwen2.5-0.5B-Instruct`

To rollback or switch active model:
```powershell
# Start server explicitly pointing to desired checkpoint:
python scripts/serve.py --checkpoint checkpoints/bravien-v2
```
Or set environment variable:
```powershell
$env:BRAVIEN_CHECKPOINT = "checkpoints/bravien-v2"
```
