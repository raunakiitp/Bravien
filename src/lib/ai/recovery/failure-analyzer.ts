/**
 * Bravien Failure Analyzer (§Phase 14).
 *
 * Deterministically analyzes execution errors, classifies failure categories,
 * and determines whether bounded alternative recovery is safe and viable.
 */

import type { FailureAnalysisInput, FailureAnalysisResult, FailureCategory } from "./types";

/**
 * Analyzes execution failure and produces deterministic failure classification.
 */
export function analyzeFailure(input: FailureAnalysisInput): FailureAnalysisResult {
  const msg = (input.errorMessage || "").toLowerCase();
  const code = (input.errorCode || "").toUpperCase();

  // 1. Security Failures (Absolute Hard Gate - NEVER retry)
  if (
    code.includes("SECURITY") ||
    code.includes("SSRF") ||
    code.includes("INJECTION") ||
    code.includes("UNAUTHORIZED") ||
    code.includes("FORBIDDEN") ||
    msg.includes("ssrf") ||
    msg.includes("forbidden") ||
    msg.includes("unauthorized") ||
    msg.includes("permission denied") ||
    msg.includes("idor") ||
    msg.includes("path traversal")
  ) {
    if (code.includes("UNAUTHORIZED") || msg.includes("unauthorized") || msg.includes("permission denied")) {
      return {
        category: "AUTHORIZATION_FAILURE",
        retryable: false,
        reason: "Operation rejected due to authorization or permission constraints.",
      };
    }
    return {
      category: "SECURITY_FAILURE",
      retryable: false,
      reason: "Operation blocked by system security guardrails.",
    };
  }

  // 2. Validation Failures (Invalid parameters/schema - NEVER retry)
  if (
    code.includes("VALIDATION") ||
    code.includes("INVALID_ARGUMENT") ||
    msg.includes("validation error") ||
    msg.includes("schema validation failed") ||
    msg.includes("invalid input syntax")
  ) {
    return {
      category: "VALIDATION_FAILURE",
      retryable: false,
      reason: "Request parameters failed schema validation.",
    };
  }

  // 3. Timeouts (Retryable once)
  if (
    code.includes("TIMEOUT") ||
    code.includes("DEADLINE_EXCEEDED") ||
    msg.includes("timeout") ||
    msg.includes("timed out") ||
    msg.includes("aborted")
  ) {
    return {
      category: "TIMEOUT",
      retryable: true,
      alternativeMode: input.executionMode === "RESEARCH" ? "DIRECT" : undefined,
      reason: "Operation timed out; eligible for single bounded fallback.",
    };
  }

  // 4. Context Overflow Failures (Retryable with compaction)
  if (
    code.includes("CONTEXT") ||
    msg.includes("context length exceeded") ||
    msg.includes("maximum context length") ||
    msg.includes("token limit")
  ) {
    return {
      category: "CONTEXT_FAILURE",
      retryable: true,
      reason: "Prompt exceeded token budget; eligible for aggressive context compaction.",
    };
  }

  // 5. Model Failures (Retryable with FAST profile or deterministic shortcut)
  if (
    code.includes("MODEL") ||
    input.modelCalled ||
    msg.includes("model unavailable") ||
    msg.includes("generation failed") ||
    msg.includes("inference error") ||
    msg.includes("cuda error")
  ) {
    return {
      category: "MODEL_FAILURE",
      retryable: true,
      reason: "Local model generation failed; fallback to FAST profile or deterministic path.",
    };
  }

  // 6. Retrieval Failures (Fallback between RAG and Web Research)
  if (
    input.ragAttempted ||
    input.webResearchAttempted ||
    code.includes("RETRIEVAL") ||
    msg.includes("retrieval failed") ||
    msg.includes("no documents found") ||
    msg.includes("search failed")
  ) {
    const alternativeMode = input.executionMode === "RESEARCH" ? "TOOL" : "RESEARCH";
    return {
      category: "RETRIEVAL_FAILURE",
      retryable: true,
      alternativeMode,
      reason: "Information retrieval failed; fallback to alternative source or direct synthesis.",
    };
  }

  // 7. Tool Failures (Fallback to DIRECT or alternate tool)
  if (
    input.failedTool ||
    code.includes("TOOL") ||
    msg.includes("tool execution failed") ||
    msg.includes("calculator error") ||
    msg.includes("syntax error in math")
  ) {
    return {
      category: "TOOL_FAILURE",
      retryable: true,
      alternativeMode: "DIRECT",
      reason: `Tool '${input.failedTool ?? "unknown"}' failed; fallback to direct response.`,
    };
  }

  // 8. Planning Failures (Fallback to simplified execution)
  if (
    input.executionMode === "PLANNED" ||
    code.includes("PLAN") ||
    msg.includes("plan execution failed") ||
    msg.includes("step failed")
  ) {
    return {
      category: "PLANNING_FAILURE",
      retryable: true,
      alternativeMode: "DIRECT",
      reason: "Multi-step plan execution encountered a failure; fallback to simplified turn.",
    };
  }

  // 9. Routing Failures
  if (code.includes("ROUTING") || msg.includes("routing failed")) {
    return {
      category: "ROUTING_FAILURE",
      retryable: true,
      alternativeMode: "DIRECT",
      reason: "Execution routing failed; fallback to default DIRECT mode.",
    };
  }

  // 10. Unknown / Default
  return {
    category: "UNKNOWN",
    retryable: false,
    reason: "Unclassified failure; refusing automatic retry.",
  };
}
