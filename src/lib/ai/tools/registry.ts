import { prisma } from "@/lib/db/prisma";
import { logger } from "@/lib/observability/logger";
import type { ToolDefinition, ToolExecutionContext, ToolResult } from "./base";
import { calculatorTool } from "./calculator";
import { searchConversationsTool } from "./conversations";
import { getDocumentContentTool, searchDocumentsTool } from "./documents";
import { searchMemoriesTool } from "./memories";
import { timeTool } from "./time";

export const BUILTIN_TOOLS: Record<string, ToolDefinition> = {
  [calculatorTool.name]: calculatorTool,
  [timeTool.name]: timeTool,
  [searchMemoriesTool.name]: searchMemoriesTool,
  [searchDocumentsTool.name]: searchDocumentsTool,
  [getDocumentContentTool.name]: getDocumentContentTool,
  [searchConversationsTool.name]: searchConversationsTool,
};

export function getTool(name: string): ToolDefinition | null {
  return BUILTIN_TOOLS[name] ?? null;
}

export function listTools(): Array<{ name: string; description: string }> {
  return Object.values(BUILTIN_TOOLS).map((t) => ({
    name: t.name,
    description: t.description,
  }));
}

/**
 * Execute a tool safely with input validation and database execution logging.
 */
export async function executeTool(
  toolName: string,
  rawInput: unknown,
  context: ToolExecutionContext,
): Promise<ToolResult> {
  const tool = getTool(toolName);
  if (!tool) {
    return {
      success: false,
      error: `Tool "${toolName}" not found.`,
      formattedOutput: `Error: Tool "${toolName}" is not available.`,
    };
  }

  const parsed = tool.schema.safeParse(rawInput);
  if (!parsed.success) {
    return {
      success: false,
      error: "Invalid tool input parameters.",
      formattedOutput: `Error: Invalid parameters for ${toolName}.`,
    };
  }

  const start = Date.now();
  let result: ToolResult;
  let status: "SUCCESS" | "FAILED" = "SUCCESS";

  try {
    result = await tool.execute(parsed.data, context);
    if (!result.success) status = "FAILED";
  } catch (err) {
    status = "FAILED";
    const errorMsg = err instanceof Error ? err.message : String(err);
    result = {
      success: false,
      error: errorMsg,
      formattedOutput: `Tool execution failed: ${errorMsg}`,
    };
  }

  const durationMs = Date.now() - start;

  // Persist tool execution record asynchronously if conversationId exists
  if (context.conversationId) {
    prisma.toolExecution
      .create({
        data: {
          conversationId: context.conversationId,
          toolName,
          input: (rawInput as object) ?? {},
          output: {
            success: result.success,
            data: (result.data as object) ?? null,
            error: result.error ?? null,
          },
          status,
          durationMs,
        },
      })
      .catch((logErr) => {
        logger.warn("tool.log_failed", {
          error: logErr instanceof Error ? logErr.message : String(logErr),
        });
      });
  }

  return result;
}

/**
 * Detect deterministic tool executions from user queries.
 * Provides high-reliability tool augmentation for Qwen2.5-0.5B-Instruct.
 */
export async function detectAndExecuteTools(
  query: string,
  context: ToolExecutionContext,
): Promise<ToolResult[]> {
  const q = query.trim();
  const lower = q.toLowerCase();
  const results: ToolResult[] = [];

  // 1. Time / Date Detection
  if (
    lower.includes("what time is it") ||
    lower.includes("current time") ||
    lower.includes("today's date") ||
    lower.includes("what is today's date") ||
    lower.includes("what is the date") ||
    lower.includes("what day is it") ||
    lower.includes("current date and time")
  ) {
    const timeRes = await executeTool(timeTool.name, {}, context);
    results.push(timeRes);
  }

  // 2. Math Expression Detection
  // e.g. "calculate 25 * 4", "what is 144 / 12", "evaluate sqrt(81) + 10"
  const mathMatch =
    q.match(/(?:calculate|compute|evaluate|what is)\s+([0-9+\-*/%^().\s\w]+=[?]?|[0-9+\-*/%^().a-zA-Z_\s]+)/i) ||
    q.match(/^([0-9+\-*/%^().\s]+[+\-*/%^][0-9+\-*/%^().\s]+)$/);

  if (mathMatch) {
    const expr = mathMatch[1].replace(/=|\?/g, "").trim();
    if (/[0-9]/.test(expr) && /[+\-*/%^()]|sqrt|sin|cos|log/.test(expr)) {
      const calcRes = await executeTool(calculatorTool.name, { expression: expr }, context);
      if (calcRes.success) {
        results.push(calcRes);
      }
    }
  }

  // 3. Document search explicit request
  if (context.projectId && (lower.startsWith("search documents for") || lower.startsWith("search files for"))) {
    const docQuery = q.replace(/^(?:search documents for|search files for)\s*/i, "").trim();
    if (docQuery) {
      const docRes = await executeTool(searchDocumentsTool.name, { query: docQuery }, context);
      results.push(docRes);
    }
  }

  // 4. Memory search explicit request
  if (lower.startsWith("search memories for") || lower.startsWith("do i remember") || lower.startsWith("what are my preferences for")) {
    const memQuery = q.replace(/^(?:search memories for|do i remember|what are my preferences for)\s*/i, "").trim();
    if (memQuery) {
      const memRes = await executeTool(searchMemoriesTool.name, { query: memQuery }, context);
      results.push(memRes);
    }
  }

  return results;
}
