/**
 * Context Minimization, Tool Compression, and Output Budgeting for Bravien.
 *
 * Optimizes prompts sent to the local 0.5B model by:
 * - Filtering out irrelevant memories and conversation history
 * - Compressing structured tool outputs to essential factual lines
 * - Dynamically sizing output token limits based on query complexity
 */

import type { AIMessage } from "@/types";

export type OutputBudgetTier = "VERY_SHORT" | "NORMAL" | "DETAILED" | "CODE";

export interface OutputBudget {
  tier: OutputBudgetTier;
  maxTokens: number;
  reason: string;
}

/**
 * Estimates the optimal maximum output tokens for a given query.
 */
export function estimateOutputBudget(
  query: string,
  options: { isCodingMode?: boolean; isSimpleFactual?: boolean } = {},
): OutputBudget {
  const norm = query.trim().toLowerCase();

  // 1. Coding queries need higher adaptive output space
  if (
    options.isCodingMode ||
    /\b(?:code|script|function|class|method|component|sql|regex|python|typescript|javascript|rust|go|c\+\+|html|css|algorithm|refactor)\b/i.test(
      norm,
    )
  ) {
    return {
      tier: "CODE",
      maxTokens: 1536,
      reason: "Programming request requiring complete code block generation",
    };
  }

  // 2. Very short queries (factual, definitions, translations, yes/no)
  if (
    options.isSimpleFactual ||
    /^(?:what is|who is|define|translate|meaning of|when did|where is)\s+[a-z0-9\s]{1,30}[?]?$/i.test(
      norm,
    ) ||
    norm.split(" ").length <= 4
  ) {
    return {
      tier: "VERY_SHORT",
      maxTokens: 128,
      reason: "Simple factual definition or short answer query",
    };
  }

  // 3. Detailed / Long-form explanations & comparisons
  if (
    /\b(?:explain in detail|comprehensive|step by step|compare and contrast|write an essay|full report|pros and cons)\b/i.test(
      norm,
    ) ||
    norm.length > 120
  ) {
    return {
      tier: "DETAILED",
      maxTokens: 640,
      reason: "Detailed analytical explanation or multi-point comparison",
    };
  }

  // 4. Default standard conversational response
  return {
    tier: "NORMAL",
    maxTokens: 280,
    reason: "Standard conversational response",
  };
}

/**
 * Filters memories to only those relevant to the current user query.
 */
export function filterRelevantMemories(memories: string[], query: string, max: number = 3): string[] {
  if (!memories || memories.length === 0) return [];
  const terms = query
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((w) => w.length > 2);

  if (terms.length === 0) {
    return memories.slice(0, max);
  }

  // Score each memory by term overlap
  const scored = memories.map((mem) => {
    const lower = mem.toLowerCase();
    let score = 0;
    for (const term of terms) {
      if (lower.includes(term)) {
        score += 2;
      }
    }
    // Boost general workflow or style preferences
    if (/\b(?:always|never|prefer|format|style|language)\b/i.test(lower)) {
      score += 1;
    }
    return { mem, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.filter((s) => s.score > 0 || memories.length <= max).slice(0, max).map((s) => s.mem);
}

/**
 * Compresses tool results by removing redundant metadata wrappers and whitespace.
 */
export function compressToolResult(raw: string): string {
  if (!raw) return "";
  let clean = raw.trim();

  // Strip JSON debug envelopes if present
  try {
    const parsed = JSON.parse(clean);
    if (parsed && typeof parsed === "object") {
      if (parsed.result !== undefined) clean = String(parsed.result);
      else if (parsed.output !== undefined) clean = String(parsed.output);
      else if (parsed.formatted !== undefined) clean = String(parsed.formatted);
    }
  } catch {
    // not json
  }

  // Remove duplicate blank lines and trim lines
  clean = clean
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .join("\n");

  return clean;
}
