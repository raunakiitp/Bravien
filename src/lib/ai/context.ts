import type { AIMessage } from "@/types";
import { BRAVIEN_IDENTITY } from "./identity";
import { buildSystemPromptDetailed } from "./prompts";

export interface BuildContextOptions {
  messages: AIMessage[];
  contextWindow?: number | null;
  userPreferences?: string | null;
  projectInstructions?: string | null;
  projectDocumentsContext?: string | null;
  toolResultsFormatted?: string | null;
  memories?: string[];
  conversationSummary?: string | null;
  isCodingMode?: boolean;
}

export interface BuiltContext {
  systemPrompt: string;
  messages: AIMessage[];
  tier: "minimal" | "compact" | "full";
  droppedItems: string[];
  estimatedTotalTokens: number;
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Builds token-budget-aware context for model inference.
 * Prioritization order:
 * 1. System instructions & Bravien identity
 * 2. Current user request
 * 3. Tool results (calculation, time, memory search, etc.)
 * 4. Relevant project documents / RAG context
 * 5. Relevant memories
 * 6. Recent conversation history (with summary fallback)
 */
export function buildContext(options: BuildContextOptions): BuiltContext {
  const contextWindow = options.contextWindow ?? 32768;
  const droppedItems: string[] = [];

  // 1. Build Base System Prompt
  const promptOptions = {
    userPreferences: options.userPreferences,
    projectInstructions: options.projectInstructions,
    projectDocumentsContext: options.projectDocumentsContext,
    memories: options.memories,
    contextWindow,
  };

  const { prompt: basePrompt, tier } = buildSystemPromptDetailed(promptOptions);

  // Append coding instructions if requested or auto-detected
  const systemPromptParts = [basePrompt];
  if (options.isCodingMode) {
    systemPromptParts.push(`\n${BRAVIEN_IDENTITY.codingGuidelines}`);
  }

  // Append Tool Results block if present
  if (options.toolResultsFormatted?.trim()) {
    systemPromptParts.push(
      `\nTool Execution Results (use these facts directly in your answer):\n${options.toolResultsFormatted.trim()}`,
    );
  }

  // Append Prior Conversation Summary if present
  if (options.conversationSummary?.trim()) {
    systemPromptParts.push(
      `\n${options.conversationSummary.trim()}`,
    );
  }

  const finalSystemPrompt = systemPromptParts.join("\n\n");
  const systemTokens = estimateTokens(finalSystemPrompt);

  // Reserve tokens for model reply (minimum 512, maximum 2048)
  const reservedOutputTokens = Math.min(2048, Math.max(512, Math.floor(contextWindow * 0.2)));
  const availableHistoryTokens = Math.max(200, contextWindow - systemTokens - reservedOutputTokens);

  // Budget conversation messages backwards from the most recent
  const budgetedMessages: AIMessage[] = [];
  let currentHistoryTokens = 0;

  const reversed = [...options.messages].reverse();
  for (const msg of reversed) {
    const msgTokens = estimateTokens(msg.content) + 4;
    if (currentHistoryTokens + msgTokens <= availableHistoryTokens || budgetedMessages.length === 0) {
      budgetedMessages.unshift(msg);
      currentHistoryTokens += msgTokens;
    } else {
      droppedItems.push(`older_turn_${msg.role}`);
    }
  }

  const estimatedTotalTokens = systemTokens + currentHistoryTokens;

  return {
    systemPrompt: finalSystemPrompt,
    messages: budgetedMessages,
    tier,
    droppedItems,
    estimatedTotalTokens,
  };
}
