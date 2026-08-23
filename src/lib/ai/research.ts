/**
 * Bounded Web Research Workflow Engine for Bravien.
 */

import { logger } from "@/lib/observability/logger";
import { defaultWebSearchProvider, type WebSearchResult } from "@/lib/research/provider";
import {
  formatEvidenceForPrompt,
  type EvidenceItem,
  extractCitations,
} from "./evidence";

export interface ResearchWorkflowOptions {
  maxSearchResults?: number;
  fetchTopPage?: boolean;
  timeoutMs?: number;
}

export interface ResearchResult {
  query: string;
  evidenceItems: EvidenceItem[];
  formattedEvidence: string;
  citations: Array<{ title: string; url: string; snippet: string }>;
  resultsCount: number;
}

/**
 * Runs a bounded web research pipeline: search -> deduplicate -> rank -> extract evidence.
 */
export async function runResearchWorkflow(
  query: string,
  options: ResearchWorkflowOptions = {},
): Promise<ResearchResult> {
  const maxResults = options.maxSearchResults ?? 4;
  const timeoutMs = options.timeoutMs ?? 8000;
  const q = query.trim();

  const evidenceItems: EvidenceItem[] = [];

  try {
    const rawResults = await defaultWebSearchProvider.search(q, {
      maxResults,
      timeoutMs,
    });

    for (let i = 0; i < rawResults.length; i++) {
      const r = rawResults[i];
      let content = r.snippet;

      // Optionally fetch full excerpt for the top result if snippet is brief
      if (i === 0 && options.fetchTopPage && r.url) {
        try {
          const pageText = await defaultWebSearchProvider.fetchPage(r.url, {
            maxContentLength: 1000,
            timeoutMs: 3000,
          });
          if (pageText && pageText.length > content.length) {
            content = pageText;
          }
        } catch {
          // Keep snippet if page fetch times out
        }
      }

      evidenceItems.push({
        sourceType: "WEB",
        sourceId: `web_${i + 1}`,
        title: r.title,
        content,
        url: r.url,
        confidence: r.score,
        timestamp: new Date().toISOString(),
        metadata: { domain: r.domain },
      });
    }
  } catch (err) {
    logger.warn("research_workflow.failed", {
      error: err instanceof Error ? err.message : String(err),
    });
  }

  const formattedEvidence = formatEvidenceForPrompt(evidenceItems);
  const citations = extractCitations(evidenceItems);

  return {
    query: q,
    evidenceItems,
    formattedEvidence,
    citations,
    resultsCount: evidenceItems.length,
  };
}
