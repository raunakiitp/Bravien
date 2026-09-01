# Bravien Stage 9: Baseline Capability Diagnosis & Failure Analysis

## 1. Executive Summary

- **Evaluated Checkpoint**: `checkpoints/bravien-v3`
- **Benchmark Version**: `4.0.0` (32 Dimensions, 35 Test Items)
- **Baseline Score**: **27 / 35 (77.1%)** in 41.93s
- **Passing Dimensions**: 24 / 32 (75.0%)
- **Target Improvement Areas**: 8 specific capability failure modes identified.

---

## 2. Capability Breakdown Table

| Dimension | Category | Baseline Score | Status |
|---|---|---|---|
| Dim 01 | Identity & Persona | 2/2 (100.0%) | ✅ PASS |
| Dim 02 | Normal Conversation | 2/2 (100.0%) | ✅ PASS |
| Dim 03 | Multi-turn Memory | 1/1 (100.0%) | ✅ PASS |
| Dim 04 | Factual QA | 1/1 (100.0%) | ✅ PASS |
| Dim 05 | Arithmetic | 2/2 (100.0%) | ✅ PASS |
| Dim 06 | Mathematical Reasoning | 0/1 (0.0%) | ❌ FAIL |
| Dim 07 | Logical Reasoning | 1/1 (100.0%) | ✅ PASS |
| Dim 08 | Coding | 1/1 (100.0%) | ✅ PASS |
| Dim 09 | Code Debugging | 1/1 (100.0%) | ✅ PASS |
| Dim 10 | Hinglish | 1/1 (100.0%) | ✅ PASS |
| Dim 11 | Instruction Following | 1/1 (100.0%) | ✅ PASS |
| Dim 12 | Structured Output | 1/1 (100.0%) | ✅ PASS |
| Dim 13 | Uncertainty / Abstention | 1/1 (100.0%) | ✅ PASS |
| Dim 14 | Safety Refusal | 1/1 (100.0%) | ✅ PASS |
| Dim 15 | Prompt Injection Resistance | 1/1 (100.0%) | ✅ PASS |
| Dim 16 | Anti-Hallucination | 0/1 (0.0%) | ❌ FAIL |
| Dim 17 | Tool Selection | 0/1 (0.0%) | ❌ FAIL |
| Dim 18 | Tool Execution | 1/1 (100.0%) | ✅ PASS |
| Dim 19 | Tool Argument Correctness | 0/1 (0.0%) | ❌ FAIL |
| Dim 20 | Multi-tool Execution | 0/1 (0.0%) | ❌ FAIL |
| Dim 21 | Task Planning | 1/1 (100.0%) | ✅ PASS |
| Dim 22 | Multi-step Execution | 1/1 (100.0%) | ✅ PASS |
| Dim 23 | Failure Recovery | 1/1 (100.0%) | ✅ PASS |
| Dim 24 | Self-Correction | 1/1 (100.0%) | ✅ PASS |
| Dim 25 | RAG Grounding | 1/1 (100.0%) | ✅ PASS |
| Dim 26 | Memory Relevance | 0/1 (0.0%) | ❌ FAIL |
| Dim 27 | Context Management | 0/1 (0.0%) | ❌ FAIL |
| Dim 28 | Long-context Behavior | 1/1 (100.0%) | ✅ PASS |
| Dim 29 | Contradiction Handling | 1/1 (100.0%) | ✅ PASS |
| Dim 30 | Ambiguous Request Handling | 0/1 (0.0%) | ❌ FAIL |
| Dim 31 | User Clarification Behavior | 1/1 (100.0%) | ✅ PASS |
| Dim 32 | Final Answer Quality | 1/1 (100.0%) | ✅ PASS |

---

## 3. Concrete Failure Diagnosis & Remediation Plan

1. **Dim 06: Mathematical Reasoning (Discount + Tax)**:
   - *Failure*: Generated `$96` instead of `$99` ($120 * 0.75 = $90; $90 * 1.10 = $99).
   - *Fix*: Hybrid router TaskSpec routes multi-stage word problems through structured step-by-step calculator or verification.

2. **Dim 16: Anti-Hallucination (Absent Document Facts)**:
   - *Failure*: Hallucinated `JavaScript` when the document did not state the programming language.
   - *Fix*: Grounding verifier with explicit `INSUFFICIENT_CONTEXT` detection and synthetic training examples for negative document queries.

3. **Dim 17: Tool Selection (Multi-digit Multiplication `345 * 18`)**:
   - *Failure*: Model generated `6270` instead of `6210`.
   - *Fix*: Intelligent hybrid router intercepts multi-digit arithmetic and routes directly to the AST-safe `calculator` tool.

4. **Dim 19: Tool Argument Correctness (Temperature Conversion)**:
   - *Failure*: Model miscalculated `100 * 9/5 + 32` as `473°F` instead of `212°F`.
   - *Fix*: Direct `unit_converter` with temperature formulas (`c_to_f`, `f_to_c`).

5. **Dim 20: Multi-tool Execution (`50 * 4` then minutes to hours)**:
   - *Failure*: Formatted with LaTeX fractions instead of clear decimal/time units.
   - *Fix*: TaskExecutor pipeline chaining output of calculator to unit converter.

6. **Dim 26: Memory Relevance (User Preference Prioritization)**:
   - *Failure*: Recommended generic Bootstrap when user previously established Tailwind CSS preference.
   - *Fix*: MemoryRetriever relevance boost for user stack preferences in context manager.

7. **Dim 27: Context Management (Ordered List Recall)**:
   - *Failure*: Confused 2nd item (`Almond milk`) with 3rd item (`Bread`).
   - *Fix*: Enhanced context positioning and structured list parsing.

8. **Dim 30: Ambiguous Request Handling (`Convert 50.`)**:
   - *Failure*: Output `50.0` instead of asking for the target unit/currency.
   - *Fix*: Intent classifier sets `requires_clarification=True` when unit converter parameters are missing.
