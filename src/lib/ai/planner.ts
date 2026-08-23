/**
 * Bounded Task Planning Engine for Bravien.
 *
 * For complex multi-step user tasks, decomposes the goal into a finite,
 * dependency-ordered sequence of verifiable steps.
 */

import { logger } from "@/lib/observability/logger";
import type { IntentResult } from "./router";
import type { ToolExecutionContext, ToolResult } from "./tools/base";
import { executeTool } from "./tools/registry";

export const MAX_PLAN_STEPS = 6;
export const MAX_TOOL_ITERATIONS = 5;
export const DEFAULT_PLAN_TIMEOUT_MS = 15_000;

export type StepStatus =
  | "PENDING"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "SKIPPED";

export interface PlanStep {
  id: string;
  stepNumber: number;
  description: string;
  toolName?: string;
  toolInput?: Record<string, unknown>;
  status: StepStatus;
  dependencies: string[];
  result?: ToolResult;
  error?: string;
}

export interface Plan {
  id: string;
  goal: string;
  steps: PlanStep[];
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";
  totalDurationMs?: number;
}

/**
 * Creates a bounded deterministic execution plan for a complex query.
 */
export function createPlan(
  query: string,
  intentResult: IntentResult,
  context: { projectId?: string | null } = {},
): Plan {
  const planId = `plan_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
  const q = query.trim();
  const lower = q.toLowerCase();
  const steps: PlanStep[] = [];

  // Comparison workflow (e.g., "Research X, compare with Y, and recommend")
  const compareMatch = q.match(/compare\s+(.+?)\s+(?:with|and|vs)\s+(.+?)(?:\.|$|,|\band\b)/i);
  if (compareMatch) {
    const itemA = compareMatch[1].trim();
    const itemB = compareMatch[2].trim();

    steps.push({
      id: "step_1",
      stepNumber: 1,
      description: `Gather information on ${itemA}`,
      toolName: "search_web",
      toolInput: { query: itemA },
      status: "PENDING",
      dependencies: [],
    });

    steps.push({
      id: "step_2",
      stepNumber: 2,
      description: `Gather information on ${itemB}`,
      toolName: "search_web",
      toolInput: { query: itemB },
      status: "PENDING",
      dependencies: [],
    });

    if (context.projectId) {
      steps.push({
        id: "step_3",
        stepNumber: 3,
        description: `Check relevant project reference documents`,
        toolName: "search_documents",
        toolInput: { query: `${itemA} ${itemB}` },
        status: "PENDING",
        dependencies: [],
      });
    }

    steps.push({
      id: "step_synthesize",
      stepNumber: steps.length + 1,
      description: `Synthesize findings and generate comparative evaluation`,
      status: "PENDING",
      dependencies: steps.map((s) => s.id),
    });
  } else if (intentResult.intent === "WEB_RESEARCH") {
    // Multi-phase web research workflow
    steps.push({
      id: "step_search",
      stepNumber: 1,
      description: `Search for recent information regarding query`,
      toolName: "search_web",
      toolInput: { query: q },
      status: "PENDING",
      dependencies: [],
    });

    steps.push({
      id: "step_summarize",
      stepNumber: 2,
      description: `Synthesize search findings into evidence-grounded summary`,
      status: "PENDING",
      dependencies: ["step_search"],
    });
  } else if (intentResult.intent === "DOCUMENT_QUERY" && context.projectId) {
    steps.push({
      id: "step_search_docs",
      stepNumber: 1,
      description: `Search project document repository`,
      toolName: "search_documents",
      toolInput: { query: q },
      status: "PENDING",
      dependencies: [],
    });

    steps.push({
      id: "step_analyze",
      stepNumber: 2,
      description: `Analyze document excerpts and extract pertinent facts`,
      status: "PENDING",
      dependencies: ["step_search_docs"],
    });
  } else {
    // Default 2-step structured goal
    steps.push({
      id: "step_1",
      stepNumber: 1,
      description: `Evaluate requirements and retrieve context for: ${q.slice(0, 80)}`,
      status: "PENDING",
      dependencies: [],
    });
    steps.push({
      id: "step_2",
      stepNumber: 2,
      description: `Synthesize comprehensive final response`,
      status: "PENDING",
      dependencies: ["step_1"],
    });
  }

  // Enforce hard upper bound on plan steps
  const cappedSteps = steps.slice(0, MAX_PLAN_STEPS).map((s, idx) => ({
    ...s,
    stepNumber: idx + 1,
  }));

  return {
    id: planId,
    goal: q,
    steps: cappedSteps,
    status: "PENDING",
  };
}

/**
 * Executes an execution plan safely with timeout and step limits.
 */
export async function executePlan(
  plan: Plan,
  context: ToolExecutionContext,
  options: { timeoutMs?: number } = {},
): Promise<{
  completedPlan: Plan;
  evidenceTexts: string[];
  summaryText: string;
}> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_PLAN_TIMEOUT_MS;
  const startTime = Date.now();
  plan.status = "RUNNING";

  const evidenceTexts: string[] = [];
  let toolCount = 0;

  for (const step of plan.steps) {
    if (Date.now() - startTime > timeoutMs) {
      step.status = "SKIPPED";
      step.error = "Execution timeout reached";
      continue;
    }

    if (step.toolName) {
      if (toolCount >= MAX_TOOL_ITERATIONS) {
        step.status = "SKIPPED";
        step.error = "Maximum tool iteration budget reached";
        continue;
      }

      toolCount++;
      step.status = "RUNNING";
      try {
        const result = await executeTool(
          step.toolName,
          step.toolInput ?? {},
          context,
        );
        step.result = result;
        step.status = result.success ? "COMPLETED" : "FAILED";
        if (result.formattedOutput) {
          evidenceTexts.push(result.formattedOutput);
        }
      } catch (err) {
        step.status = "FAILED";
        step.error = err instanceof Error ? err.message : String(err);
        logger.warn("planner.step_failed", {
          stepId: step.id,
          error: step.error,
        });
      }
    } else {
      step.status = "COMPLETED";
    }
  }

  const allCompleted = plan.steps.every(
    (s) => s.status === "COMPLETED" || s.status === "SKIPPED",
  );
  plan.status = allCompleted ? "COMPLETED" : "FAILED";
  plan.totalDurationMs = Date.now() - startTime;

  const summaryText = plan.steps
    .map((s) => `Step ${s.stepNumber} [${s.status}]: ${s.description}${s.error ? ` (${s.error})` : ""}`)
    .join("\n");

  return {
    completedPlan: plan,
    evidenceTexts,
    summaryText,
  };
}
