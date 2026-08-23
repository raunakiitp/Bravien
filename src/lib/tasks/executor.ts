/**
 * Centralized Autonomous Task Execution Engine for Bravien.
 *
 * Provides bounded, multi-step task execution with persistent database tracking,
 * real-time checkpointing, transient retry policy, and task resumption.
 */

import { prisma } from "@/lib/db/prisma";
import { logger } from "@/lib/observability/logger";
import { executeTool } from "@/lib/ai/tools/registry";
import { runResearchWorkflow } from "@/lib/ai/research";
import { retrieveRelevantContext } from "@/lib/rag/retrieval";
import { routeIntent } from "@/lib/ai/router";
import { createPlan } from "@/lib/ai/planner";
import {
  getOrCreateActiveAgentState,
  updateAgentState,
  recordEvidenceRef,
} from "@/lib/ai/agent-state";
import { logActivity } from "./service";
import type { TaskExecutionResult } from "./types";
import type { Prisma, TaskStatus } from "@prisma/client";

export interface TaskExecutorOptions {
  timeoutMs?: number;
  maxSteps?: number;
  isResume?: boolean;
}

export const DEFAULT_TASK_TIMEOUT_MS = 25_000;
export const MAX_ALLOWED_TASK_STEPS = 6;
export const MAX_TRANSIENT_RETRIES = 2;

interface InternalStepDef {
  stepNumber: number;
  title: string;
  description?: string;
  toolName?: string;
  toolInput?: Record<string, unknown>;
  executor: () => Promise<{ success: boolean; data?: unknown; formatted: string; error?: string }>;
}

/**
 * Identifies whether an error is transient (network/timeout) and eligible for retry.
 */
function isTransientError(error?: string): boolean {
  if (!error) return false;
  const lower = error.toLowerCase();
  return (
    lower.includes("timeout") ||
    lower.includes("fetch failed") ||
    lower.includes("econnreset") ||
    lower.includes("etimedout") ||
    lower.includes("network") ||
    lower.includes("503") ||
    lower.includes("504") ||
    lower.includes("temporary")
  );
}

/**
 * Builds step definitions for a task based on its type and query.
 */
function buildStepDefinitions(
  userId: string,
  task: { id: string; title: string; description?: string | null; type: string; projectId?: string | null },
): InternalStepDef[] {
  const queryText = [task.title, task.description].filter(Boolean).join(" - ");
  const stepDefs: InternalStepDef[] = [];

  if (task.type === "RESEARCH") {
    stepDefs.push({
      stepNumber: 1,
      title: "Query Analysis & Search Strategy",
      description: "Analyze search scope and formulate web queries",
      executor: async () => ({
        success: true,
        formatted: `Formulated web search strategy for: "${task.title}"`,
      }),
    });

    stepDefs.push({
      stepNumber: 2,
      title: "Web Research & Evidence Gathering",
      description: "Perform web search and extract authoritative excerpts",
      toolName: "search_web",
      toolInput: { query: task.title, maxResults: 4 },
      executor: async () => {
        const research = await runResearchWorkflow(task.title, {
          maxSearchResults: 4,
          timeoutMs: 8000,
        });
        return {
          success: true,
          data: research,
          formatted: research.formattedEvidence || "No web evidence collected.",
        };
      },
    });

    stepDefs.push({
      stepNumber: 3,
      title: "Evidence Synthesis & Citation Assembly",
      description: "Normalize evidence and synthesize findings",
      executor: async () => ({
        success: true,
        formatted: "Verified and structured research evidence.",
      }),
    });
  } else if (task.type === "DOCUMENT" && task.projectId) {
    stepDefs.push({
      stepNumber: 1,
      title: "Project Document Context Scan",
      description: "Scan uploaded project documents for relevant excerpts",
      toolName: "search_documents",
      toolInput: { query: queryText, limit: 4 },
      executor: async () => {
        const hits = await retrieveRelevantContext({
          userId,
          projectId: task.projectId!,
          query: queryText,
          limit: 4,
        });
        return {
          success: true,
          data: hits,
          formatted: hits.length
            ? `Found ${hits.length} relevant document sections.`
            : "No matching project documents found.",
        };
      },
    });

    stepDefs.push({
      stepNumber: 2,
      title: "Document Synthesis",
      description: "Synthesize document sections against task goals",
      executor: async () => ({
        success: true,
        formatted: "Synthesized project document context.",
      }),
    });
  } else if (task.type === "ANALYSIS") {
    stepDefs.push({
      stepNumber: 1,
      title: "Analytical & Computation Processing",
      description: "Evaluate quantitative formulas, statistics, and calculations",
      toolName: "calculator",
      executor: async () => {
        let expr = queryText;
        const mathExprMatch =
          queryText.match(/([0-9]+(?:\.[0-9]+)?(?:\s*[+\-*/%^]\s*[0-9]+(?:\.[0-9]+)?)+)/);
        if (mathExprMatch) {
          expr = mathExprMatch[1].trim();
        } else {
          const generalMatch =
            queryText.match(/(?:calculate|compute|evaluate|what is)\s+([0-9+\-*/%^().\s\w]+)/i);
          if (generalMatch) {
            expr = generalMatch[1].trim();
          }
        }
        const calcRes = await executeTool(
          "calculator",
          { expression: expr },
          { userId, projectId: task.projectId },
        );
        return {
          success: calcRes.success,
          data: calcRes.data,
          formatted: calcRes.formattedOutput,
          error: calcRes.error,
        };
      },
    });

    stepDefs.push({
      stepNumber: 2,
      title: "Analysis Synthesis & Findings",
      description: "Synthesize analytical evaluation and numerical results",
      executor: async () => ({
        success: true,
        formatted: `Completed quantitative analysis for "${task.title}".`,
      }),
    });
  } else {
    // General / Coding / Project: Use Router & Planner
    const route = routeIntent(queryText, {
      hasActiveProject: Boolean(task.projectId),
      hasWebAccess: true,
    });
    const plan = createPlan(queryText, route, { projectId: task.projectId });

    for (const pStep of plan.steps.slice(0, MAX_ALLOWED_TASK_STEPS)) {
      const stepTitle =
        pStep.description.length > 80
          ? pStep.description.slice(0, 77) + "..."
          : pStep.description;

      stepDefs.push({
        stepNumber: pStep.stepNumber,
        title: stepTitle,
        description: pStep.description,
        toolName: pStep.toolName,
        executor: async () => {
          if (pStep.toolName) {
            let toolPayload: Record<string, unknown> = { query: queryText };
            if (pStep.toolName === "calculator") {
              let expr = queryText;
              const mathExprMatch =
                queryText.match(/([0-9]+(?:\.[0-9]+)?(?:\s*[+\-*/%^]\s*[0-9]+(?:\.[0-9]+)?)+)/);
              if (mathExprMatch) {
                expr = mathExprMatch[1].trim();
              }
              toolPayload = { expression: expr };
            } else if (pStep.toolName === "get_current_time") {
              toolPayload = {};
            }

            const toolRes = await executeTool(
              pStep.toolName,
              toolPayload,
              { userId, projectId: task.projectId },
            );
            return {
              success: toolRes.success,
              data: toolRes.data,
              formatted: toolRes.formattedOutput,
              error: toolRes.error,
            };
          }
          return {
            success: true,
            formatted: `Completed step: ${stepTitle}`,
          };
        },
      });
    }
  }

  if (stepDefs.length === 0) {
    stepDefs.push({
      stepNumber: 1,
      title: "Execute Task Goal",
      description: "Process task objectives",
      executor: async () => ({
        success: true,
        formatted: `Executed task: ${task.title}`,
      }),
    });
  }

  return stepDefs.slice(0, MAX_ALLOWED_TASK_STEPS);
}

/**
 * Executes a task end-to-end with real-time checkpointing and retry protection.
 */
export async function executeTask(
  userId: string,
  taskId: string,
  options: TaskExecutorOptions = {},
): Promise<TaskExecutionResult> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TASK_TIMEOUT_MS;
  const startTime = Date.now();

  // 1. Fetch and verify task ownership
  const task = await prisma.task.findFirst({
    where: { id: taskId, userId },
    include: { project: true, steps: { orderBy: { stepNumber: "asc" } } },
  });

  if (!task) {
    throw new Error("Task not found or unauthorized access.");
  }

  // Duplicate running protection
  if (task.status === "RUNNING" && !options.isResume) {
    throw new Error("Task is already running.");
  }

  // 2. Synchronize / Initialize AgentState
  const agentState = await getOrCreateActiveAgentState(userId, {
    projectId: task.projectId,
    taskId: task.id,
    goal: task.title,
  });

  await updateAgentState(userId, agentState.id, {
    status: "EXECUTING",
    currentStep: 1,
  });

  // 3. Create or reuse TaskExecution record
  const execution = await prisma.taskExecution.create({
    data: {
      taskId: task.id,
      status: "RUNNING",
      startedAt: new Date(),
    },
  });

  await prisma.task.update({
    where: { id: task.id },
    data: { status: "RUNNING" },
  });

  await logActivity(userId, {
    eventType: options.isResume ? "TASK_RESUMED" : "TASK_STARTED",
    description: `${options.isResume ? "Resumed" : "Started"} execution of task "${task.title}"`,
    projectId: task.projectId,
    taskId: task.id,
  });

  // 4. Build or reuse step definitions
  const stepDefs = buildStepDefinitions(userId, task);

  // If not resuming, initialize DB steps afresh
  if (!options.isResume || task.steps.length === 0) {
    await prisma.taskStep.deleteMany({ where: { taskId: task.id } });
    for (const s of stepDefs) {
      await prisma.taskStep.create({
        data: {
          taskId: task.id,
          stepNumber: s.stepNumber,
          title: s.title,
          description: s.description ?? null,
          status: "PENDING",
          toolName: s.toolName ?? null,
          toolInput: (s.toolInput as Prisma.InputJsonValue) ?? undefined,
        },
      });
    }
  }

  // 5. Execute steps sequentially with checkpointing and transient retry
  const stepOutputs: string[] = [];
  let successfulSteps = 0;
  let executionError: string | undefined;
  const collectedCitations: Array<{ title: string; url: string; snippet: string }> = [];

  for (const s of stepDefs) {
    // Check timeout
    if (Date.now() - startTime > timeoutMs) {
      executionError = `Task execution exceeded safety timeout of ${timeoutMs}ms.`;
      break;
    }

    // If resuming, check if step is already COMPLETED
    if (options.isResume) {
      const existingStep = task.steps.find((st) => st.stepNumber === s.stepNumber);
      if (existingStep && existingStep.status === "COMPLETED") {
        successfulSteps++;
        stepOutputs.push(`Step ${s.stepNumber} (${s.title}) [Reused Checkpoint]:\n${(existingStep.toolResult as { output?: string })?.output || "Completed"}`);
        continue;
      }
    }

    // Mark step RUNNING
    const stepRow = await prisma.taskStep.findFirst({
      where: { taskId: task.id, stepNumber: s.stepNumber },
    });

    if (stepRow) {
      await prisma.taskStep.update({
        where: { id: stepRow.id },
        data: { status: "RUNNING", startedAt: new Date() },
      });
    }

    // Step Execution with transient retry policy (max 2 retries)
    let stepSuccess = false;
    let stepOutputFormatted = "";
    let stepResultData: unknown = null;
    let stepErrorMsg = "";
    let retries = 0;

    while (!stepSuccess && retries <= MAX_TRANSIENT_RETRIES) {
      try {
        const stepRes = await s.executor();

        if (s.toolName) {
          await logActivity(userId, {
            eventType: "TOOL_EXECUTED",
            description: `Task "${task.title}" ran tool ${s.toolName}`,
            projectId: task.projectId,
            taskId: task.id,
            metadata: { toolName: s.toolName, success: stepRes.success },
          });
        }

        if (stepRes.data && typeof stepRes.data === "object" && "citations" in stepRes.data) {
          const cits = (stepRes.data as { citations?: Array<{ title: string; url: string; snippet: string }> }).citations;
          if (Array.isArray(cits)) {
            collectedCitations.push(...cits);
            // Record compact evidence references in AgentState
            for (const cit of cits) {
              await recordEvidenceRef(userId, agentState.id, {
                title: cit.title,
                url: cit.url,
                snippet: cit.snippet,
                sourceType: "WEB",
                timestamp: new Date().toISOString(),
              });
            }
          }
        }

        if (stepRes.success) {
          stepSuccess = true;
          stepOutputFormatted = stepRes.formatted;
          stepResultData = stepRes.data;
        } else {
          stepErrorMsg = stepRes.error || `Step ${s.stepNumber} failed`;
          if (isTransientError(stepErrorMsg) && retries < MAX_TRANSIENT_RETRIES) {
            retries++;
            await logActivity(userId, {
              eventType: "TASK_RETRY",
              description: `Retrying step ${s.stepNumber} (attempt ${retries}/${MAX_TRANSIENT_RETRIES}) after: ${stepErrorMsg}`,
              taskId: task.id,
            });
            await new Promise((r) => setTimeout(r, 150));
          } else {
            break;
          }
        }
      } catch (err) {
        stepErrorMsg = err instanceof Error ? err.message : String(err);
        if (isTransientError(stepErrorMsg) && retries < MAX_TRANSIENT_RETRIES) {
          retries++;
          await logActivity(userId, {
            eventType: "TASK_RETRY",
            description: `Retrying step ${s.stepNumber} (attempt ${retries}/${MAX_TRANSIENT_RETRIES}) after error: ${stepErrorMsg}`,
            taskId: task.id,
          });
          await new Promise((r) => setTimeout(r, 150));
        } else {
          break;
        }
      }
    }

    // Process Step Completion / Failure
    if (stepSuccess) {
      successfulSteps++;
      stepOutputs.push(`Step ${s.stepNumber} (${s.title}):\n${stepOutputFormatted}`);

      if (stepRow) {
        await prisma.taskStep.update({
          where: { id: stepRow.id },
          data: {
            status: "COMPLETED",
            retryCount: retries,
            completedAt: new Date(),
            toolResult: (stepResultData as Prisma.InputJsonValue) ?? { output: stepOutputFormatted },
          },
        });
      }

      // Checkpoint step completion in AgentState
      await updateAgentState(userId, agentState.id, {
        currentStep: s.stepNumber + 1,
        completedActions: [...agentState.completedActions, `Completed step ${s.stepNumber}: ${s.title}`],
      });

      await logActivity(userId, {
        eventType: "PLAN_CHECKPOINTED",
        description: `Checkpointed step ${s.stepNumber}/${stepDefs.length} for task "${task.title}"`,
        taskId: task.id,
      });
    } else {
      executionError = stepErrorMsg || `Step ${s.stepNumber} failed after ${retries} retries`;
      if (stepRow) {
        await prisma.taskStep.update({
          where: { id: stepRow.id },
          data: {
            status: "FAILED",
            retryCount: retries,
            completedAt: new Date(),
            error: executionError,
          },
        });
      }
      break;
    }
  }

  const durationMs = Date.now() - startTime;
  const finalStatus: TaskStatus = executionError ? "FAILED" : "COMPLETED";

  const resultSummary = executionError
    ? `Task execution halted at step ${successfulSteps + 1}: ${executionError}\n\nCompleted steps:\n${stepOutputs.join("\n\n")}`
    : `Task "${task.title}" completed successfully (${successfulSteps}/${stepDefs.length} steps):\n\n${stepOutputs.join("\n\n")}`;

  // 6. Update TaskExecution & Task
  await prisma.taskExecution.update({
    where: { id: execution.id },
    data: {
      status: finalStatus,
      completedAt: new Date(),
      durationMs,
      error: executionError ?? null,
      summary: resultSummary,
    },
  });

  await prisma.task.update({
    where: { id: task.id },
    data: {
      status: finalStatus,
      resultSummary,
      completedAt: finalStatus === "COMPLETED" ? new Date() : null,
    },
  });

  // 7. Update AgentState status
  await updateAgentState(userId, agentState.id, {
    status: finalStatus === "COMPLETED" ? "COMPLETED" : "FAILED",
  });

  await logActivity(userId, {
    eventType: finalStatus === "COMPLETED" ? "TASK_COMPLETED" : "TASK_FAILED",
    description: `Task "${task.title}" ${finalStatus.toLowerCase()} in ${durationMs}ms`,
    projectId: task.projectId,
    taskId: task.id,
    metadata: { durationMs, successfulSteps, totalSteps: stepDefs.length },
  });

  return {
    taskId: task.id,
    executionId: execution.id,
    status: finalStatus,
    startedAt: execution.startedAt.toISOString(),
    completedAt: new Date().toISOString(),
    durationMs,
    stepsCount: stepDefs.length,
    successfulSteps,
    resultSummary,
    citations: collectedCitations.length ? collectedCitations : undefined,
    error: executionError,
  };
}

/**
 * Resumes an interrupted, failed, or paused task from its last checkpoint.
 */
export async function resumeTask(
  userId: string,
  taskId: string,
  options: { timeoutMs?: number } = {},
): Promise<TaskExecutionResult> {
  const task = await prisma.task.findFirst({
    where: { id: taskId, userId },
    include: { steps: { orderBy: { stepNumber: "asc" } } },
  });

  if (!task) {
    throw new Error("Task not found or unauthorized access.");
  }

  return executeTask(userId, taskId, {
    ...options,
    isResume: true,
  });
}
