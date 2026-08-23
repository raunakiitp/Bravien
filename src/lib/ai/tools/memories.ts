import { z } from "zod";
import { searchMemories } from "@/lib/memory/service";
import type { ToolDefinition } from "./base";

const searchMemoriesSchema = z.object({
  query: z.string().min(1, "Search query is required"),
  limit: z.number().int().min(1).max(20).optional().default(5),
});

export const searchMemoriesTool: ToolDefinition<typeof searchMemoriesSchema> = {
  name: "search_memories",
  description:
    "Search through the user's stored persistent memories for preferences, profile facts, workflows, or project instructions.",
  schema: searchMemoriesSchema,
  execute: async ({ query, limit }, context) => {
    try {
      const results = await searchMemories(context.userId, query, {
        projectId: context.projectId,
        limit,
      });

      if (results.length === 0) {
        return {
          success: true,
          data: [],
          formattedOutput: `No stored memories found matching query "${query}".`,
        };
      }

      const formatted = results
        .map((m, i) => `[${i + 1}] (${m.type}): ${m.content}`)
        .join("\n");

      return {
        success: true,
        data: results,
        formattedOutput: `Found ${results.length} relevant memories:\n${formatted}`,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
        formattedOutput: `Failed to search memories: ${err instanceof Error ? err.message : "Error"}`,
      };
    }
  },
};
