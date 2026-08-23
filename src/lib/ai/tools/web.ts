import { z } from "zod";
import type { ToolDefinition, ToolResult } from "./base";
import { defaultWebSearchProvider, isSafeUrl } from "@/lib/research/provider";

export const searchWebInputSchema = z.object({
  query: z.string().min(1, "Query cannot be empty").max(300),
  maxResults: z.number().int().min(1).max(8).optional().default(4),
});

export const searchWebTool: ToolDefinition<typeof searchWebInputSchema> = {
  name: "search_web",
  description: "Searches the public web for real-time information, news, documentation, or facts.",
  category: "WEB",
  riskLevel: "LOW",
  requiresNetwork: true,
  requiresProjectScope: false,
  mutatesData: false,
  executionTimeoutMs: 8000,
  schema: searchWebInputSchema,
  execute: async (input, _context): Promise<ToolResult> => {
    try {
      const results = await defaultWebSearchProvider.search(input.query, {
        maxResults: input.maxResults,
      });

      if (!results.length) {
        return {
          success: true,
          data: { query: input.query, results: [] },
          formattedOutput: `No web results found for "${input.query}".`,
        };
      }

      const formatted = results
        .map(
          (r, idx) =>
            `[Source ${idx + 1}: ${r.title}]\nURL: ${r.url}\nDomain: ${r.domain}\nExcerpt: ${r.snippet}`,
        )
        .join("\n\n---\n\n");

      return {
        success: true,
        data: { query: input.query, results },
        formattedOutput: `Web Search Results for "${input.query}":\n\n${formatted}`,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
        formattedOutput: `Web search failed: ${err instanceof Error ? err.message : "Error"}`,
      };
    }
  },
};

export const fetchWebPageInputSchema = z.object({
  url: z.string().url("Invalid URL format").max(500),
});

export const fetchWebPageTool: ToolDefinition<typeof fetchWebPageInputSchema> = {
  name: "fetch_web_page",
  description: "Fetches and extracts clean readable text from a safe public HTTP/HTTPS URL.",
  category: "WEB",
  riskLevel: "MEDIUM",
  requiresNetwork: true,
  requiresProjectScope: false,
  mutatesData: false,
  executionTimeoutMs: 8000,
  schema: fetchWebPageInputSchema,
  execute: async (input, _context): Promise<ToolResult> => {
    if (!isSafeUrl(input.url)) {
      return {
        success: false,
        error: "URL rejected: Access to private networks, localhost, and non-HTTP protocols is forbidden.",
        formattedOutput: `Fetch failed: ${input.url} is not a permitted public URL.`,
      };
    }

    try {
      const content = await defaultWebSearchProvider.fetchPage(input.url, {
        maxContentLength: 1500,
      });

      if (!content) {
        return {
          success: false,
          error: "Could not retrieve content from the specified URL.",
          formattedOutput: `Could not retrieve content from ${input.url}.`,
        };
      }

      return {
        success: true,
        data: { url: input.url, content },
        formattedOutput: `Content from ${input.url}:\n\n${content}`,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
        formattedOutput: `Fetch error for ${input.url}: ${err instanceof Error ? err.message : "Error"}`,
      };
    }
  },
};
