/**
 * Canonical Model Prompt Formatter & Generation Configuration for Bravien.
 *
 * Centralizes prompt formatting, token budget limits, generation profiles,
 * and priority-based graceful context reduction tailored specifically for local
 * 0.5B-scale instruction-tuned models (e.g. Qwen2.5-0.5B / Bravien).
 */

import { buildContext, type BuildContextOptions } from "./context";
import type { AIMessage } from "@/types";

export type GenerationProfile = "FAST" | "BALANCED" | "CREATIVE" | "CODE";

export interface GenerationSettings {
  temperature: number;
  topP: number;
  topK?: number;
  repetitionPenalty: number;
  maxTokens: number;
  stop?: string[];
}

export const GENERATION_PROFILES: Record<GenerationProfile, GenerationSettings> = {
  FAST: {
    temperature: 0.1,
    topP: 0.85,
    repetitionPenalty: 1.05,
    maxTokens: 512,
    stop: ["<|endoftext|>", "<|im_end|>", "</ASSISTANT>"],
  },
  BALANCED: {
    temperature: 0.4,
    topP: 0.9,
    repetitionPenalty: 1.1,
    maxTokens: 1024,
    stop: ["<|endoftext|>", "<|im_end|>", "</ASSISTANT>"],
  },
  CREATIVE: {
    temperature: 0.7,
    topP: 0.95,
    repetitionPenalty: 1.15,
    maxTokens: 1536,
    stop: ["<|endoftext|>", "<|im_end|>", "</ASSISTANT>"],
  },
  CODE: {
    temperature: 0.1,
    topP: 0.85,
    repetitionPenalty: 1.05,
    maxTokens: 2048,
    stop: ["<|endoftext|>", "<|im_end|>", "</ASSISTANT>"],
  },
};

/**
 * Strict token budgeting limits for local 0.5B inference.
 */
export const MODEL_BUDGET_LIMITS = {
  MAX_CONVERSATION_TURNS: 8,
  MAX_MEMORIES: 5,
  MAX_RAG_CHUNKS: 4,
  MAX_WEB_SOURCES: 6,
  MAX_AGENT_STATE_CHARS: 300,
  DEFAULT_MAX_CONTEXT_TOKENS: 2048,
  RESERVED_OUTPUT_TOKENS: 512,
} as const;

export interface ModelPromptOptions extends BuildContextOptions {
  profile?: GenerationProfile;
}

/**
 * Fast estimation of prompt tokens (approx. 4 characters per token).
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

/**
 * Get generation configuration parameters for a specified profile.
 */
export function getGenerationConfig(profile: GenerationProfile = "BALANCED"): GenerationSettings {
  return GENERATION_PROFILES[profile] || GENERATION_PROFILES.BALANCED;
}

/**
 * Formats structured messages into the final bounded prompt for the model runtime.
 * Implements strict priority-based degradation if prompt exceeds token limits:
 * 1. Drop older conversation turns (preserving the latest user request)
 * 2. Reduce web evidence
 * 3. Reduce RAG document context
 * 4. Reduce memories
 * 5. Compact agent state
 * 6. Preserves current user request and core system rules unconditionally
 */
export function formatModelPrompt(options: ModelPromptOptions): {
  systemPrompt: string;
  messages: AIMessage[];
  fullPromptMessages: AIMessage[];
  generationSettings: GenerationSettings;
  budgetReport: {
    estimatedInputTokens: number;
    budgetLimit: number;
    degradations: string[];
  };
} {
  const profile = options.profile ?? (options.isCodingMode ? "CODE" : "BALANCED");
  const generationSettings = getGenerationConfig(profile);
  const degradations: string[] = [];

  const contextWindow = options.contextWindow ?? MODEL_BUDGET_LIMITS.DEFAULT_MAX_CONTEXT_TOKENS;
  const reservedOutput = Math.min(generationSettings.maxTokens, MODEL_BUDGET_LIMITS.RESERVED_OUTPUT_TOKENS);
  const maxInputBudget = Math.max(256, contextWindow - reservedOutput);

  // 1. Initial bounds
  let rawMessages = (options.messages || []).filter((m) => m.role !== "system");
  let boundedMessages = rawMessages.slice(-MODEL_BUDGET_LIMITS.MAX_CONVERSATION_TURNS);
  if (rawMessages.length > boundedMessages.length) {
    degradations.push("trimmed_excess_turns");
  }
  let boundedMemories = (options.memories || []).slice(0, MODEL_BUDGET_LIMITS.MAX_MEMORIES);
  let evidence = options.evidenceFormatted || "";
  let ragContext = options.projectDocumentsContext || "";
  let agentState = options.agentStateSummary ? options.agentStateSummary.slice(0, MODEL_BUDGET_LIMITS.MAX_AGENT_STATE_CHARS) : "";

  // Step 1 check & degrade conversation history (preserving last turn)
  let built = buildContext({
    ...options,
    messages: boundedMessages,
    memories: boundedMemories,
    evidenceFormatted: evidence,
    projectDocumentsContext: ragContext,
    agentStateSummary: agentState,
    contextWindow,
  });

  let totalTokens = built.estimatedTotalTokens;

  // Step 1: Drop older turns if exceeding budget
  while (totalTokens > maxInputBudget && boundedMessages.length > 1) {
    boundedMessages = boundedMessages.slice(1);
    degradations.push("dropped_older_turn");
    built = buildContext({
      ...options,
      messages: boundedMessages,
      memories: boundedMemories,
      evidenceFormatted: evidence,
      projectDocumentsContext: ragContext,
      agentStateSummary: agentState,
      contextWindow,
    });
    totalTokens = built.estimatedTotalTokens;
  }

  // Step 2: Reduce web evidence if still over budget
  if (totalTokens > maxInputBudget && evidence.length > 300) {
    evidence = evidence.slice(0, 300) + "\n...[truncated web evidence]";
    degradations.push("reduced_web_evidence");
    built = buildContext({
      ...options,
      messages: boundedMessages,
      memories: boundedMemories,
      evidenceFormatted: evidence,
      projectDocumentsContext: ragContext,
      agentStateSummary: agentState,
      contextWindow,
    });
    totalTokens = built.estimatedTotalTokens;
  }

  // Step 3: Reduce RAG chunks if still over budget
  if (totalTokens > maxInputBudget && ragContext.length > 300) {
    ragContext = ragContext.slice(0, 300) + "\n...[truncated document context]";
    degradations.push("reduced_rag_context");
    built = buildContext({
      ...options,
      messages: boundedMessages,
      memories: boundedMemories,
      evidenceFormatted: evidence,
      projectDocumentsContext: ragContext,
      agentStateSummary: agentState,
      contextWindow,
    });
    totalTokens = built.estimatedTotalTokens;
  }

  // Step 4: Reduce memories if still over budget
  if (totalTokens > maxInputBudget && boundedMemories.length > 2) {
    boundedMemories = boundedMemories.slice(0, 2);
    degradations.push("reduced_memories");
    built = buildContext({
      ...options,
      messages: boundedMessages,
      memories: boundedMemories,
      evidenceFormatted: evidence,
      projectDocumentsContext: ragContext,
      agentStateSummary: agentState,
      contextWindow,
    });
    totalTokens = built.estimatedTotalTokens;
  }

  // Step 5: Compact agent state
  if (totalTokens > maxInputBudget && agentState.length > 100) {
    agentState = agentState.slice(0, 100);
    degradations.push("compacted_agent_state");
    built = buildContext({
      ...options,
      messages: boundedMessages,
      memories: boundedMemories,
      evidenceFormatted: evidence,
      projectDocumentsContext: ragContext,
      agentStateSummary: agentState,
      contextWindow,
    });
    totalTokens = built.estimatedTotalTokens;
  }

  const fullPromptMessages: AIMessage[] = [
    { role: "system", content: built.systemPrompt },
    ...built.messages,
  ];

  return {
    systemPrompt: built.systemPrompt,
    messages: built.messages,
    fullPromptMessages,
    generationSettings,
    budgetReport: {
      estimatedInputTokens: built.estimatedTotalTokens,
      budgetLimit: maxInputBudget,
      degradations,
    },
  };
}
