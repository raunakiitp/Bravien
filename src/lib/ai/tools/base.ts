import { z } from "zod";

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
  schema: TSchema;
  execute: (
    input: z.infer<TSchema>,
    context: ToolExecutionContext,
  ) => Promise<ToolResult>;
}
