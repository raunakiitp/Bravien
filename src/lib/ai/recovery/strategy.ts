/**
 * Bravien Recovery Strategy Engine (§Phase 14).
 *
 * Deterministically maps classified failure modes to bounded alternative
 * execution strategies with strict token budgets and single-retry limits.
 */

import type { FailureAnalysisResult, RecoveryStrategy } from "./types";

export interface RecoveryStrategyOptions {
  currentMode: string;
  userQuery: string;
  hasActiveProject?: boolean;
  hasWebAccess?: boolean;
  isCodingMode?: boolean;
  attemptCount?: number;
}

/**
 * Maps failure analysis to a concrete, bounded alternative strategy.
 */
export function selectRecoveryStrategy(
  failure: FailureAnalysisResult,
  options: RecoveryStrategyOptions,
): RecoveryStrategy | null {
  // Hard limit: Maximum 1 recovery attempt per request
  if ((options.attemptCount ?? 0) >= 1) {
    return null;
  }

  // Non-retryable failure categories
  if (!failure.retryable) {
    return null;
  }

  const { currentMode, isCodingMode } = options;

  switch (failure.category) {
    case "MODEL_FAILURE":
      return {
        id: "strat_fast_fallback",
        description: "Retry with bounded FAST generation profile and strict token limit",
        targetMode: currentMode,
        fallbackProfile: "FAST",
        maxTokens: isCodingMode ? 1536 : 280,
      };

    case "CONTEXT_FAILURE":
      return {
        id: "strat_compact_context",
        description: "Retry with aggressive context compaction and minimal history",
        targetMode: currentMode,
        fallbackProfile: "FAST",
        maxTokens: 280,
      };

    case "TOOL_FAILURE":
      return {
        id: "strat_tool_to_direct",
        description: "Bypass failing tool and respond directly with factual synthesis",
        targetMode: "DIRECT",
        fallbackProfile: "FAST",
        maxTokens: 128,
        useDeterministicShortcut: true,
      };

    case "RETRIEVAL_FAILURE":
      if (currentMode === "RESEARCH" && options.hasActiveProject) {
        return {
          id: "strat_research_to_rag",
          description: "Fallback from web research to local project document retrieval",
          targetMode: "TOOL",
          fallbackProfile: "BALANCED",
          maxTokens: 280,
        };
      }
      if (currentMode === "TOOL" && options.hasWebAccess) {
        return {
          id: "strat_rag_to_research",
          description: "Fallback from project documents to public web research",
          targetMode: "RESEARCH",
          fallbackProfile: "BALANCED",
          maxTokens: 280,
        };
      }
      return {
        id: "strat_retrieval_direct",
        description: "Fallback to direct synthesis with available context",
        targetMode: "DIRECT",
        fallbackProfile: "FAST",
        maxTokens: 128,
      };

    case "PLANNING_FAILURE":
      return {
        id: "strat_plan_to_direct",
        description: "Simplify multi-step plan to single-turn direct reasoning",
        targetMode: "DIRECT",
        fallbackProfile: "BALANCED",
        maxTokens: 640,
      };

    case "TIMEOUT":
      return {
        id: "strat_timeout_fast",
        description: "Retry with fast profile and small token budget",
        targetMode: "DIRECT",
        fallbackProfile: "FAST",
        maxTokens: 128,
      };

    case "ROUTING_FAILURE":
      return {
        id: "strat_routing_fallback",
        description: "Fallback to safe DIRECT execution mode",
        targetMode: "DIRECT",
        fallbackProfile: "FAST",
        maxTokens: 128,
      };

    default:
      return null;
  }
}
