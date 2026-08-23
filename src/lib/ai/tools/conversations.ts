import { z } from "zod";
import { searchConversations } from "@/lib/search/conversations";
import type { ToolDefinition } from "./base";

const searchConversationsSchema = z.object({
  query: z.string().min(1, "Search query is required"),
  limit: z.number().int().min(1).max(20).optional().default(5),
});

export const searchConversationsTool: ToolDefinition<typeof searchConversationsSchema> = {
  name: "search_conversations",
  description: "Search previous conversations and message history for past discussions or decisions.",
  category: "CONVERSATION",
  riskLevel: "LOW",
  requiresNetwork: false,
  requiresProjectScope: false,
  mutatesData: false,
  executionTimeoutMs: 3000,
  schema: searchConversationsSchema,
  execute: async ({ query, limit }, context) => {
    try {
      const results = await searchConversations(context.userId, query, { limit });

      if (results.length === 0) {
        return {
          success: true,
          data: [],
          formattedOutput: `No past conversations found matching "${query}".`,
        };
      }

      const formatted = results
        .map(
          (r, i) =>
            `[${i + 1}] "${r.title}" (${new Date(r.updatedAt).toLocaleDateString()}): ${r.snippet ? `"...${r.snippet}..."` : "Title match"}`,
        )
        .join("\n");

      return {
        success: true,
        data: results,
        formattedOutput: `Found ${results.length} relevant past conversations:\n${formatted}`,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
        formattedOutput: `Failed to search conversations: ${err instanceof Error ? err.message : "Error"}`,
      };
    }
  },
};
