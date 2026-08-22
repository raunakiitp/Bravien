import { z } from "zod";
import { getFeatureFlags } from "@/lib/features";
import { calculatorTool } from "@/lib/tools/calculator";
import { webSearchTool } from "@/lib/tools/web-search";

export interface ToolDefinition<TParams extends z.ZodType = z.ZodType> {
  name: string;
  description: string;
  parameters: TParams;
  execute: (args: unknown) => Promise<unknown>;
}

const TOOLS: ToolDefinition[] = [calculatorTool, webSearchTool];

export function getTool(name: string): ToolDefinition | undefined {
  return TOOLS.find((t) => t.name === name);
}

/** Tools enabled by feature flags / env configuration. */
export function getEnabledTools(): ToolDefinition[] {
  const flags = getFeatureFlags();
  return TOOLS.filter((tool) => {
    if (tool.name === "web_search") return flags.web_search;
    return true;
  });
}

export async function executeTool(
  name: string,
  args: unknown,
): Promise<unknown> {
  const tool = getTool(name);
  if (!tool) {
    return { error: `Unknown tool: ${name}` };
  }
  const enabled = getEnabledTools().some((t) => t.name === name);
  if (!enabled) {
    return { error: `Tool disabled: ${name}` };
  }
  return tool.execute(args);
}

/** OpenAI-compatible tool schemas for chat completions. */
export function toOpenAITools(
  tools: ToolDefinition[] = getEnabledTools(),
): Array<{
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}> {
  return tools.map((tool) => {
    const jsonSchema = z.toJSONSchema(tool.parameters) as Record<
      string,
      unknown
    >;
    const { $schema, ...parameters } = jsonSchema;
    void $schema;
    return {
      type: "function" as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters,
      },
    };
  });
}
