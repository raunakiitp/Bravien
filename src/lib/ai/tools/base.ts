import { z } from "zod";

export type ToolCategory =
  | "SYSTEM"
  | "MATH"
  | "MEMORY"
  | "DOCUMENT"
  | "WEB"
  | "CONVERSATION";

export type ToolRiskLevel = "LOW" | "MEDIUM" | "HIGH";

export interface ToolExecutionContext {
  userId: string;
  projectId?: string | null;
  conversationId?: string | null;
}

export interface ToolResult<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  formattedOutput: string;
}

export interface ToolDefinition<TSchema extends z.ZodType = z.ZodType> {
  name: string;
  description: string;
  category: ToolCategory;
  riskLevel: ToolRiskLevel;
  schema: TSchema;
  executionTimeoutMs?: number;
  requiresNetwork?: boolean;
  requiresProjectScope?: boolean;
  mutatesData?: boolean;
  execute: (
    input: z.infer<TSchema>,
    context: ToolExecutionContext,
  ) => Promise<ToolResult>;
}
