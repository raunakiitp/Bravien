/**
 * Intelligent Inference Gating for Bravien.
 *
 * Deterministically determines whether a user query can be answered directly
 * without invoking local LLM inference, minimizing token usage, latency,
 * and unnecessary GPU/CPU workload.
 */

import { routeIntent, type IntentResult } from "./router";

export type GatingMode = "DETERMINISTIC" | "MODEL_REQUIRED" | "HYBRID";

export interface InferenceGateDecision {
  shouldGenerate: boolean;
  reason: string;
  mode: GatingMode;
  confidence: number;
  deterministicTool?: string;
  directResponse?: string;
}

export interface GateContext {
  hasActiveProject?: boolean;
  hasWebAccess?: boolean;
  isCodingMode?: boolean;
}

const DETERMINISTIC_IDENTITY_PATTERNS = [
  /^(?:who are you|what is your name|what are you called|introduce yourself)[?]?$/i,
  /^(?:what can you do|what are your capabilities|what tools do you have)[?]?$/i,
  /^(?:ping|hello|hi|hey|test)[.!]?$/i,
];

const SIMPLE_TASK_STATUS_PATTERNS = [
  /^(?:task status|status of task|get task status|check task)\s*([a-z0-9_-]+)?[?]?$/i,
];

/**
 * Evaluates a user query to determine whether local model generation is required.
 */
export function evaluateInferenceGate(
  query: string,
  context: GateContext = {},
): InferenceGateDecision {
  const norm = query.trim();
  const lower = norm.toLowerCase();

  // 1. Check known system/identity shortcuts
  for (const pattern of DETERMINISTIC_IDENTITY_PATTERNS) {
    if (pattern.test(lower)) {
      if (/^(?:ping|test)[.!]?$/i.test(lower)) {
        return {
          shouldGenerate: false,
          mode: "DETERMINISTIC",
          reason: "Simple ping/test heartbeat query",
          confidence: 1.0,
          directResponse: "Pong! Bravien runtime is active and operational.",
        };
      }
      if (/^(?:hello|hi|hey)[.!]?$/i.test(lower)) {
        return {
          shouldGenerate: false,
          mode: "DETERMINISTIC",
          reason: "Deterministic greeting shortcut",
          confidence: 1.0,
          directResponse: "Hello! I am Bravien, your local AI assistant. How can I help you today?",
        };
      }
      if (/^(?:who are you|what is your name|what are you called|introduce yourself)[?]?$/i.test(lower)) {
        return {
          shouldGenerate: false,
          mode: "DETERMINISTIC",
          reason: "Canonical identity response shortcut",
          confidence: 1.0,
          directResponse:
            "I am Bravien, an independent local-first AI assistant powered by local inference.",
        };
      }
      if (/^(?:what can you do|what are your capabilities|what tools do you have)[?]?$/i.test(lower)) {
        return {
          shouldGenerate: false,
          mode: "DETERMINISTIC",
          reason: "System capabilities response shortcut",
          confidence: 1.0,
          directResponse:
            "I can assist with conversation, writing, mathematical calculations, time/date queries, persistent memory management, document RAG search, web research, autonomous task execution, and coding.",
        };
      }
    }
  }

  // 2. Intent-based gating
  const intent = routeIntent(norm, {
    hasActiveProject: context.hasActiveProject ?? false,
    hasWebAccess: context.hasWebAccess ?? true,
  });

  // Pure Math Calculation
  if (intent.intent === "CALCULATION") {
    return {
      shouldGenerate: false,
      mode: "DETERMINISTIC",
      reason: "Mathematical expressions are completely resolved by deterministic calculator",
      confidence: intent.confidence,
      deterministicTool: "calculator",
    };
  }

  // Pure Time & Date
  if (intent.intent === "TIME_DATE") {
    return {
      shouldGenerate: false,
      mode: "DETERMINISTIC",
      reason: "Current time/date is completely resolved by deterministic time tool",
      confidence: intent.confidence,
      deterministicTool: "get_current_time",
    };
  }

  // Task Status Query
  for (const pattern of SIMPLE_TASK_STATUS_PATTERNS) {
    if (pattern.test(lower)) {
      return {
        shouldGenerate: false,
        mode: "DETERMINISTIC",
        reason: "Task status is retrieved directly from database state",
        confidence: 0.9,
        deterministicTool: "get_task_status",
      };
    }
  }

  // Hybrid Mode: Tool / Web / Document retrieval requiring model synthesis
  if (
    intent.requiresWeb ||
    intent.intent === "WEB_RESEARCH" ||
    intent.intent === "DOCUMENT_QUERY" ||
    intent.intent === "COMPARISON" ||
    intent.requiresPlanning
  ) {
    return {
      shouldGenerate: true,
      mode: "HYBRID",
      reason: "Requires tool/research evidence retrieval followed by natural language synthesis",
      confidence: intent.confidence,
    };
  }

  // Model-Required: Coding, Reasoning, Creative, Open-ended chat
  return {
    shouldGenerate: true,
    mode: "MODEL_REQUIRED",
    reason: "Open-ended natural language generation, reasoning, or programming request",
    confidence: intent.confidence,
  };
}
