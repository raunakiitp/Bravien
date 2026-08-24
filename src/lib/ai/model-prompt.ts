/**
 * Canonical Model Prompt Formatter & Generation Configuration for Bravien.
 *
 * Centralizes prompt formatting, token budget limits, and generation profiles
 * tailored specifically for local 0.5B-scale instruction-tuned models (e.g. Qwen2.5-0.5B / Bravien).
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
 * Get generation configuration parameters for a specified profile.
 */
export function getGenerationConfig(profile: GenerationProfile = "BALANCED"): GenerationSettings {
  return GENERATION_PROFILES[profile] || GENERATION_PROFILES.BALANCED;
}

/**
 * Formats structured messages into the final bounded prompt for the model runtime.
 */
export function formatModelPrompt(options: ModelPromptOptions): {
  systemPrompt: string;
  messages: AIMessage[];
  fullPromptMessages: AIMessage[];
  generationSettings: GenerationSettings;
} {
  const profile = options.profile ?? (options.isCodingMode ? "CODE" : "BALANCED");
  const generationSettings = getGenerationConfig(profile);

  // Bound conversation turns to prevent context blowup on local models
  const boundedMessages = (options.messages || [])
    .filter((m) => m.role !== "system")
    .slice(-MODEL_BUDGET_LIMITS.MAX_CONVERSATION_TURNS);

  // Bound memories
  const boundedMemories = (options.memories || []).slice(0, MODEL_BUDGET_LIMITS.MAX_MEMORIES);

  const contextWindow = options.contextWindow ?? MODEL_BUDGET_LIMITS.DEFAULT_MAX_CONTEXT_TOKENS;

  const built = buildContext({
    ...options,
    messages: boundedMessages,
    memories: boundedMemories,
    contextWindow,
  });

  const fullPromptMessages: AIMessage[] = [
    { role: "system", content: built.systemPrompt },
    ...built.messages,
  ];

  return {
    systemPrompt: built.systemPrompt,
    messages: built.messages,
    fullPromptMessages,
    generationSettings,
  };
}

/**
 * Fast estimation of prompt tokens (approx. 4 characters per token).
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}
