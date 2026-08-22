import { z } from "zod";

const webSearchParams = z.object({
  query: z.string().min(1).describe("Search query"),
  max_results: z.number().int().min(1).max(10).optional(),
});

export interface WebSearchResult {
  unavailable?: boolean;
  message?: string;
  query?: string;
  results?: Array<{
    title: string;
    url: string;
    content?: string;
    score?: number;
  }>;
}

/**
 * Tavily-compatible web search via POST https://api.tavily.com/search
 * Requires SEARCH_API_KEY. Returns a soft unavailable payload when missing.
 */
export async function runWebSearch(args: {
  query: string;
  max_results?: number;
}): Promise<WebSearchResult> {
  const apiKey = process.env.SEARCH_API_KEY?.trim();
  if (!apiKey) {
    return {
      unavailable: true,
      message:
        "Web search is not configured. Set SEARCH_API_KEY (Tavily) to enable.",
      query: args.query,
    };
  }

  const response = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      api_key: apiKey,
      query: args.query,
      max_results: args.max_results ?? 5,
      include_answer: false,
      search_depth: "basic",
    }),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    return {
      unavailable: true,
      message: `Search API error (${response.status}): ${text.slice(0, 300)}`,
      query: args.query,
    };
  }

  const json = (await response.json()) as {
    results?: Array<{
      title?: string;
      url?: string;
      content?: string;
      score?: number;
    }>;
  };

  return {
    query: args.query,
    results: (json.results ?? []).map((r) => ({
      title: r.title ?? "",
      url: r.url ?? "",
      content: r.content,
      score: r.score,
    })),
  };
}

export const webSearchTool = {
  name: "web_search" as const,
  description:
    "Search the public web for up-to-date information. Returns titles, URLs, and snippets.",
  parameters: webSearchParams,
  async execute(args: unknown): Promise<WebSearchResult> {
    const parsed = webSearchParams.parse(args);
    return runWebSearch(parsed);
  },
};
