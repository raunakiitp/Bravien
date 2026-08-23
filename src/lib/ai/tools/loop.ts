import { logger } from "@/lib/observability/logger";
import type { ToolExecutionContext, ToolResult } from "./base";
import { executeTool } from "./registry";
import { calculatorTool } from "./calculator";
import { searchConversationsTool } from "./conversations";
import { getDocumentContentTool, searchDocumentsTool } from "./documents";
import { searchMemoriesTool } from "./memories";
import { timeTool } from "./time";

export interface ToolLoopOptions {
  maxIterations?: number;
  timeoutMs?: number;
}

export interface ToolExecutionSummary {
  executedTools: Array<{
    name: string;
    input: unknown;
    result: ToolResult;
    durationMs: number;
  }>;
  combinedFormattedOutput: string;
  iterationCount: number;
}

/**
 * Controlled multi-tool execution loop with iteration limits, timeout protection,
 * and graceful failure handling.
 */
export async function runToolLoop(
  query: string,
  context: ToolExecutionContext,
  options: ToolLoopOptions = {},
): Promise<ToolExecutionSummary> {
  const maxIterations = options.maxIterations ?? 3;
  const timeoutMs = options.timeoutMs ?? 10_000;
  const startTime = Date.now();

  const executedTools: ToolExecutionSummary["executedTools"] = [];
  const q = query.trim();
  const lower = q.toLowerCase();

  let iteration = 0;

  // Plan tools needed based on user query
  const toolsToRun: Array<{ name: string; input: unknown }> = [];

  // 1. Time / Date query
  if (
    lower.includes("what time is it") ||
    lower.includes("current time") ||
    lower.includes("today's date") ||
    lower.includes("what is today's date") ||
    lower.includes("what is the date") ||
    lower.includes("what day is it") ||
    lower.includes("current date and time")
  ) {
    toolsToRun.push({ name: timeTool.name, input: {} });
  }

  // 2. Math calculation query
  const mathMatch =
    q.match(/(?:calculate|compute|evaluate|what is)\s+([0-9+\-*/%^().\s\w]+=[?]?|[0-9+\-*/%^().a-zA-Z_\s]+)/i) ||
    q.match(/^([0-9+\-*/%^().\s]+[+\-*/%^][0-9+\-*/%^().\s]+)$/);

  if (mathMatch) {
    const expr = mathMatch[1].replace(/=|\?/g, "").trim();
    if (/[0-9]/.test(expr) && /[+\-*/%^()]|sqrt|sin|cos|log/.test(expr)) {
      toolsToRun.push({ name: calculatorTool.name, input: { expression: expr } });
    }
  }

  // 3. Document search
  if (
    context.projectId &&
    (lower.startsWith("search documents for") ||
      lower.startsWith("search files for") ||
      lower.includes("find in project documents"))
  ) {
    const docQuery = q
      .replace(/^(?:search documents for|search files for|find in project documents)\s*/i, "")
      .trim();
    if (docQuery) {
      toolsToRun.push({ name: searchDocumentsTool.name, input: { query: docQuery } });
    }
  }

  // 4. Memory search
  if (
    lower.startsWith("search memories for") ||
    lower.startsWith("do i remember") ||
    lower.startsWith("what are my preferences for") ||
    lower.includes("check my memories for")
  ) {
    const memQuery = q
      .replace(/^(?:search memories for|do i remember|what are my preferences for|check my memories for)\s*/i, "")
      .trim();
    if (memQuery) {
      toolsToRun.push({ name: searchMemoriesTool.name, input: { query: memQuery } });
    }
  }

  // 5. Conversation history search
  if (lower.startsWith("search conversations for") || lower.startsWith("find in past chats")) {
    const chatQuery = q
      .replace(/^(?:search conversations for|find in past chats)\s*/i, "")
      .trim();
    if (chatQuery) {
      toolsToRun.push({ name: searchConversationsTool.name, input: { query: chatQuery } });
    }
  }

  // Execute planned tools within iteration cap and timeout boundary
  for (const item of toolsToRun) {
    if (iteration >= maxIterations) {
      logger.warn("tool_loop.max_iterations_reached", { maxIterations });
      break;
    }
    if (Date.now() - startTime > timeoutMs) {
      logger.warn("tool_loop.timeout_exceeded", { timeoutMs });
      break;
    }

    iteration++;
    const stepStart = Date.now();
    try {
      const result = await executeTool(item.name, item.input, context);
      executedTools.push({
        name: item.name,
        input: item.input,
        result,
        durationMs: Date.now() - stepStart,
      });
    } catch (err) {
      executedTools.push({
        name: item.name,
        input: item.input,
        result: {
          success: false,
          error: err instanceof Error ? err.message : String(err),
          formattedOutput: `Tool ${item.name} failed: ${err instanceof Error ? err.message : "Error"}`,
        },
        durationMs: Date.now() - stepStart,
      });
    }
  }

  const combinedFormattedOutput = executedTools
    .map((t) => t.result.formattedOutput)
    .filter(Boolean)
    .join("\n\n");

  return {
    executedTools,
    combinedFormattedOutput,
    iterationCount: iteration,
  };
}
