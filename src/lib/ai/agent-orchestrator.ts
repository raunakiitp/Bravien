/**
 * Unified Agent Orchestrator for Bravien.
 *
 * Coordinates the full agent execution lifecycle:
 * Request -> Router -> Mode Selection -> Memory/State -> Bounded Execution
 * (Direct, Tool, Research, Plan, Task, Confirmation) -> Evidence ->
 * Model Inference -> Checkpoint -> Safe Memory Promotion -> Observability.
 */

import { isFeatureEnabled } from "@/lib/features";
import { logger } from "@/lib/observability/logger";
import { routeIntent, type IntentResult } from "./router";
import { createPlan, executePlan } from "./planner";
import { runToolLoop } from "./tools/loop";
import { runResearchWorkflow } from "./research";
import { retrieveRelevantContext, formatRetrievedContext } from "@/lib/rag/retrieval";
import {
  getRelevantAgentState,
  summarizeAgentStateForContext,
  addConstraint,
  recordDecision,
  updateAgentState,
  transitionState,
} from "./agent-state";
import { classifyActionRisk } from "./action-proposal";
import { promoteStateToPersistentMemory } from "./memory-promotion";
import { streamChat } from "./orchestrator";
import { evaluateInferenceGate, type InferenceGateDecision } from "./inference-gate";
import { inferenceCache } from "./inference-cache";
import { efficiencyTracker } from "./efficiency";
import { filterRelevantMemories, compressToolResult, estimateOutputBudget } from "./context-optimizer";
import { analyzeFailure, selectRecoveryStrategy, strategyMemory } from "./recovery";
import type { AIMessage, AIStreamChunk, AgentEventType } from "@/types";

export type ExecutionMode =
  | "DIRECT"
  | "TOOL"
  | "RESEARCH"
  | "PLANNED"
  | "TASK"
  | "WAITING_CONFIRMATION";

export interface UnifiedAgentTurnOptions {
  userId?: string;
  projectId?: string | null;
  conversationId?: string | null;
  messages: AIMessage[];
  modelId?: string;
  signal?: AbortSignal;
  userPreferences?: string | null;
  projectInstructions?: string | null;
  memories?: string[];
  maxContextTokens?: number;
}

export interface ExecutionModeDecision {
  mode: ExecutionMode;
  reason: string;
  intentResult: IntentResult;
  candidateTool?: string;
}

/**
 * Determines the cheapest valid execution mode for a user turn.
 */
export function determineExecutionMode(
  query: string,
  options: {
    hasActiveProject?: boolean;
    hasWebAccess?: boolean;
    isTask?: boolean;
  } = {},
): ExecutionModeDecision {
  const norm = query.trim();

  // 1. High-risk actions require confirmation
  const risk = classifyActionRisk(norm);
  if (risk.requiresConfirmation) {
    return {
      mode: "WAITING_CONFIRMATION",
      reason: "Action classified as high risk or destructive",
      intentResult: {
        intent: "GENERAL_CHAT",
        confidence: 1.0,
        requiresPlanning: false,
        requiresWeb: false,
        candidateTools: [],
        entities: {},
        reasoning: "High-risk operation requires human confirmation gate",
      },
    };
  }

  // 2. Intent Routing
  const intentResult = routeIntent(norm, {
    hasActiveProject: options.hasActiveProject ?? false,
    hasWebAccess: options.hasWebAccess ?? true,
  });

  // 3. Task / Complex Multi-step Plan
  if (options.isTask || intentResult.requiresPlanning) {
    return {
      mode: "PLANNED",
      reason: "Multi-step complex request requires bounded planning",
      intentResult,
    };
  }

  // 4. Web Research Mode
  if (intentResult.requiresWeb || intentResult.intent === "WEB_RESEARCH") {
    return {
      mode: "RESEARCH",
      reason: "Request requires current web research and verification",
      intentResult,
    };
  }

  // 5. Deterministic Tool Mode (Math, Time, Document, Memory)
  if (
    intentResult.intent === "CALCULATION" ||
    intentResult.intent === "TIME_DATE" ||
    intentResult.intent === "DOCUMENT_QUERY" ||
    intentResult.intent === "MEMORY_QUERY" ||
    intentResult.candidateTools.length > 0
  ) {
    return {
      mode: "TOOL",
      reason: `Direct deterministic tool invocation for ${intentResult.intent}`,
      intentResult,
      candidateTool: intentResult.candidateTools[0],
    };
  }

  // 6. Direct conversational mode (Cheapest)
  return {
    mode: "DIRECT",
    reason: "Standard conversational turn without external tools",
    intentResult,
  };
}

/**
 * Unified Agent Orchestration Generator.
 * Streams structured execution lifecycle events alongside model tokens.
 */
export async function* runUnifiedAgentTurn(
  options: UnifiedAgentTurnOptions,
): AsyncIterable<AIStreamChunk> {
  const userId = options.userId;
  const projectId = options.projectId;
  const conversationId = options.conversationId;
  const lastUserMsg = options.messages.filter((m) => m.role === "user").slice(-1)[0];
  const userText = lastUserMsg?.content || "";

  // 1. Lifecycle Event: Agent Started
  yield {
    kind: "agent_event",
    eventType: "agent_started",
    message: "Initializing Bravien agent session",
    metadata: { userId: userId ? "authenticated" : "guest", projectId },
  };

  // 2. Load Agent State
  let activeAgentState: any = null;
  let agentStateSummary: string | null = null;
  if (userId) {
    try {
      activeAgentState = await getRelevantAgentState(userId, { projectId });
      if (activeAgentState) {
        agentStateSummary = summarizeAgentStateForContext(activeAgentState);
        yield {
          kind: "agent_event",
          eventType: "state_loaded",
          message: `Loaded agent state (${activeAgentState.status})`,
          metadata: { stateId: activeAgentState.id, status: activeAgentState.status },
        };
      }
    } catch {
      // safe fallback
    }
  }

  // 3. Inference Gating & Routing
  const hasWeb = isFeatureEnabled("web_search");
  const gate = evaluateInferenceGate(userText, {
    hasActiveProject: Boolean(projectId),
    hasWebAccess: hasWeb,
  });

  const decision = determineExecutionMode(userText, {
    hasActiveProject: Boolean(projectId),
    hasWebAccess: hasWeb,
  });

  // Direct identity/capability shortcut
  if (gate.mode === "DETERMINISTIC" && gate.directResponse) {
    yield {
      kind: "agent_event",
      eventType: "agent_started",
      message: "Processing query via direct deterministic shortcut",
    };
    yield {
      kind: "agent_event",
      eventType: "routing",
      message: `Direct shortcut: ${gate.reason}`,
      metadata: { mode: "DIRECT", reason: gate.reason },
    };
    yield {
      kind: "content_delta",
      delta: gate.directResponse,
    };
    yield {
      kind: "agent_event",
      eventType: "agent_completed",
      message: "Direct response completed",
      metadata: { mode: "DIRECT" },
    };
    efficiencyTracker.recordRequest({ avoidedModel: true, isDeterministic: true });
    return;
  }

  // Detect Coding Mode & Generation Profile
  const isCodingMode = Boolean(
    decision.intentResult.intent === "CODING" ||
      decision.intentResult.intent === "DEBUGGING" ||
      /\b(?:code|script|function|class|method|component|sql|regex|python|typescript|javascript|html|css|cpp|rust|golang|algorithm|bug|error|refactor)\b/i.test(
        userText,
      ),
  );

  const profile = isCodingMode
    ? "CODE"
    : decision.mode === "TOOL" || decision.mode === "DIRECT"
      ? "FAST"
      : "BALANCED";

  // Check Response Cache for identical safe requests
  const isCacheable = inferenceCache.isCacheable(userText, {
    requiresWeb: decision.intentResult.requiresWeb,
    isAction: decision.mode === "WAITING_CONFIRMATION",
  });

  const cacheKey = userId
    ? inferenceCache.generateKey({
        userId,
        projectId,
        query: userText,
        profile,
      })
    : null;

  if (isCacheable && cacheKey && userId) {
    const cachedResponse = inferenceCache.get(cacheKey, userId);
    if (cachedResponse) {
      yield {
        kind: "agent_event",
        eventType: "agent_started",
        message: "Serving response from cache",
      };
      yield {
        kind: "agent_event",
        eventType: "routing",
        message: "Cache hit: reusing verified local response",
        metadata: { mode: decision.mode },
      };
      yield {
        kind: "content_delta",
        delta: cachedResponse,
      };
      yield {
        kind: "agent_event",
        eventType: "agent_completed",
        message: "Cached turn completed",
        metadata: { mode: decision.mode },
      };
      efficiencyTracker.recordRequest({ avoidedModel: true, isCacheHit: true });
      return;
    }
  }

  yield {
    kind: "agent_event",
    eventType: "routing",
    message: `Selected execution mode: ${decision.mode}`,
    metadata: { mode: decision.mode, reason: decision.reason, intent: decision.intentResult.intent },
  };

  // 4. Mode-Specific Execution
  let toolResultsFormatted: string | null = null;
  let evidenceFormatted: string | null = null;
  let planSummary: string | null = null;
  let retrievedContext: string | null = null;
  const citations: Array<{ title: string; url: string; snippet?: string }> = [];

  try {
    if (decision.mode === "WAITING_CONFIRMATION") {
      yield {
        kind: "agent_event",
        eventType: "confirmation_required",
        message: "High-risk operation detected. Waiting for user confirmation.",
        metadata: { action: userText.slice(0, 100) },
      };

      if (userId && activeAgentState) {
        await transitionState(userId, activeAgentState.id, "WAITING_CONFIRMATION").catch(() => {});
      }

      yield {
        kind: "content_delta",
        delta: `⚠️ **Confirmation Required**: This action was classified as high-risk or destructive.\n\nPlease approve or reject this proposal in the Tasks & Workspace tab before proceeding.`,
      };
      return;
    }

    if (decision.mode === "PLANNED") {
      yield {
        kind: "agent_event",
        eventType: "planning",
        message: "Generating bounded multi-step execution plan",
      };

      const plan = createPlan(userText, decision.intentResult, { projectId });
      yield {
        kind: "agent_event",
        eventType: "plan_created",
        message: `Created plan with ${plan.steps.length} steps`,
        metadata: { stepsCount: plan.steps.length },
      };

      const planExec = await executePlan(
        plan,
        { userId: userId ?? "guest", projectId, conversationId },
        { timeoutMs: 12_000 },
      );

      planSummary = planExec.summaryText;
      if (planExec.evidenceTexts.length > 0) {
        evidenceFormatted = planExec.evidenceTexts.join("\n\n");
      }

      const completedCount = planExec.completedPlan.steps.filter((s) => s.status === "COMPLETED").length;
      yield {
        kind: "agent_event",
        eventType: "step_completed",
        message: `Plan executed successfully (${completedCount}/${plan.steps.length} steps)`,
      };
    } else if (decision.mode === "RESEARCH") {
      yield {
        kind: "agent_event",
        eventType: "research_started",
        message: "Gathering authoritative web research and facts",
      };

      const research = await runResearchWorkflow(userText, {
        maxSearchResults: 3,
        timeoutMs: 8000,
      });

      if (research.citations.length > 0) {
        yield {
          kind: "agent_event",
          eventType: "source_found",
          message: `Gathered ${research.citations.length} verified web sources`,
          metadata: { count: research.citations.length },
        };

        for (const cit of research.citations) {
          citations.push(cit);
          yield {
            kind: "citation",
            title: cit.title,
            url: cit.url,
            snippet: cit.snippet,
          };
        }
      }

      if (research.formattedEvidence) {
        evidenceFormatted = research.formattedEvidence;
        yield {
          kind: "agent_event",
          eventType: "evidence_added",
          message: "Normalized research evidence for model inference",
        };
      }
    } else if (decision.mode === "TOOL") {
      yield {
        kind: "agent_event",
        eventType: "tool_started",
        message: `Invoking tool handler for ${decision.intentResult.intent}`,
      };

      // Document query check
      if (decision.intentResult.intent === "DOCUMENT_QUERY" && projectId && userId) {
        const hits = await retrieveRelevantContext({
          userId,
          projectId,
          query: userText,
          limit: 4,
        });

        if (hits.length > 0) {
          retrievedContext = formatRetrievedContext(hits);
          for (const chunk of hits) {
            yield {
              kind: "citation",
              title: chunk.filename,
              url: `#${chunk.filename}-p${chunk.chunkIndex + 1}`,
              snippet: chunk.content.slice(0, 150) + "...",
            };
          }
        }
      } else {
        // Direct tool loop (calculator, time, memory search)
        const summary = await runToolLoop(
          userText,
          { userId: userId ?? "guest", projectId, conversationId },
          { maxIterations: 3, timeoutMs: 8000 },
        );
        if (summary.combinedFormattedOutput) {
          toolResultsFormatted = compressToolResult(summary.combinedFormattedOutput);
        }
      }

      yield {
        kind: "agent_event",
        eventType: "tool_completed",
        message: "Tool execution finished successfully",
      };
    }
  } catch (err) {
    logger.warn("agent_orchestrator.execution_warning", {
      mode: decision.mode,
      error: err instanceof Error ? err.message : String(err),
    });
  }

  // Filter memories to only relevant context
  const filteredMemories = filterRelevantMemories(options.memories ?? [], userText, 3);
  const outputBudget = estimateOutputBudget(userText, { isCodingMode });

  // 6. Inference Stream with Failure Analysis & Single Recovery Retry
  let assistantText = "";
  let executionSucceeded = false;

  try {
    for await (const chunk of streamChat({
      messages: options.messages,
      modelId: options.modelId,
      signal: options.signal,
      userPreferences: options.userPreferences,
      projectInstructions: options.projectInstructions,
      projectDocumentsContext: retrievedContext,
      agentStateSummary,
      toolResultsFormatted,
      evidenceFormatted,
      planSummary,
      isCodingMode,
      memories: filteredMemories,
      maxContextTokens: options.maxContextTokens ?? outputBudget.maxTokens,
      profile,
    })) {
      if (chunk.kind === "content_delta") {
        assistantText += chunk.delta;
      }
      yield chunk;
    }
    executionSucceeded = true;
  } catch (err) {
    logger.warn("agent_orchestrator.inference_failed", {
      mode: decision.mode,
      error: err instanceof Error ? err.message : String(err),
    });

    const failure = analyzeFailure({
      executionMode: decision.mode,
      errorMessage: err instanceof Error ? err.message : String(err),
      modelCalled: true,
      webResearchAttempted: decision.mode === "RESEARCH",
      ragAttempted: decision.intentResult.intent === "DOCUMENT_QUERY",
    });

    if (failure.retryable) {
      const strat = selectRecoveryStrategy(failure, {
        currentMode: decision.mode,
        userQuery: userText,
        hasActiveProject: Boolean(projectId),
        hasWebAccess: hasWeb,
        isCodingMode,
        attemptCount: 0,
      });

      if (strat) {
        yield {
          kind: "agent_event",
          eventType: "recovery_started",
          message: "Attempting bounded recovery strategy",
          metadata: { failureCategory: failure.category, strategyId: strat.id },
        };
        yield {
          kind: "agent_event",
          eventType: "recovery_strategy_selected",
          message: `Selected recovery strategy: ${strat.description}`,
          metadata: { from: decision.mode, to: strat.targetMode, strategyId: strat.id },
        };

        const tRecStart = Date.now();
        let recSuccess = false;
        try {
          if (toolResultsFormatted && strat.useDeterministicShortcut) {
            yield {
              kind: "content_delta",
              delta: toolResultsFormatted,
            };
            assistantText = toolResultsFormatted;
            recSuccess = true;
          } else {
            for await (const chunk of streamChat({
              messages: options.messages.slice(-2),
              modelId: options.modelId,
              signal: options.signal,
              userPreferences: options.userPreferences,
              projectInstructions: options.projectInstructions,
              projectDocumentsContext: retrievedContext,
              agentStateSummary,
              toolResultsFormatted,
              evidenceFormatted,
              planSummary,
              isCodingMode,
              memories: [],
              maxContextTokens: strat.maxTokens,
              profile: strat.fallbackProfile ?? "FAST",
            })) {
              if (chunk.kind === "content_delta") {
                assistantText += chunk.delta;
              }
              yield chunk;
            }
            recSuccess = Boolean(assistantText.trim());
          }

          if (recSuccess) {
            executionSucceeded = true;
            yield {
              kind: "agent_event",
              eventType: "recovery_completed",
              message: "Recovery succeeded",
              metadata: { strategyId: strat.id },
            };
            efficiencyTracker.recordRecovery({
              category: failure.category,
              strategyId: strat.id,
              success: true,
              latencyMs: Date.now() - tRecStart,
            });
            strategyMemory.recordStrategy({
              queryFingerprint: strategyMemory.generateFingerprint(userText),
              category: decision.intentResult.intent,
              successfulMode: strat.targetMode,
              userId: userId ?? undefined,
              projectId: projectId ?? undefined,
              timestamp: Date.now(),
            });
          }
        } catch (recErr) {
          yield {
            kind: "agent_event",
            eventType: "recovery_failed",
            message: "Recovery strategy failed; terminating turn safely",
            metadata: { error: recErr instanceof Error ? recErr.message : String(recErr) },
          };
          efficiencyTracker.recordRecovery({
            category: failure.category,
            strategyId: strat.id,
            success: false,
            latencyMs: Date.now() - tRecStart,
          });
          yield {
            kind: "content_delta",
            delta: "I encountered an issue processing your request and was unable to recover. Please try again or rephrase your query.",
          };
        }
      }
    } else {
      yield {
        kind: "agent_event",
        eventType: "agent_failed",
        message: `Execution failed: ${failure.reason}`,
        metadata: { category: failure.category },
      };
      yield {
        kind: "content_delta",
        delta: `Execution stopped: ${failure.reason}`,
      };
    }
  }

  // Cache successful responses for safe queries
  if (executionSucceeded && isCacheable && cacheKey && userId && assistantText.trim()) {
    inferenceCache.set(cacheKey, assistantText, { userId, projectId });
  }

  efficiencyTracker.recordRequest({
    avoidedModel: false,
    outputTokens: assistantText.length ? Math.ceil(assistantText.length / 4) : 0,
  });

  // 7. Post-Turn State Synchronization & Checkpoint
  if (userId && activeAgentState) {
    try {
      if (/\b(?:must|never|always|do not|don't|rule|constraint)\b/i.test(userText)) {
        const rule = userText.split("\n")[0].slice(0, 150);
        await addConstraint(userId, activeAgentState.id, rule);
      }
      if (assistantText && /\b(?:decided|agreed|chosen|selected|implemented)\b/i.test(assistantText)) {
        const dec = assistantText.split("\n")[0].slice(0, 150);
        await recordDecision(userId, activeAgentState.id, dec);
      }

      yield {
        kind: "agent_event",
        eventType: "checkpoint_saved",
        message: "Synchronized agent state checkpoint",
        metadata: { stateId: activeAgentState.id },
      };
    } catch {
      // safe ignore
    }
  }

  // 8. Lifecycle Event: Agent Completed
  yield {
    kind: "agent_event",
    eventType: executionSucceeded ? "agent_completed" : "agent_failed",
    message: executionSucceeded ? "Agent turn completed successfully" : "Agent turn ended with failure",
    metadata: { mode: decision.mode, success: executionSucceeded },
  };
}
