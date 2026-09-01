/**
 * Abstract Model Provider Contract for TypeScript AI Subsystem.
 */

import type { AIMessage, AIStreamChunk } from "@/types";

export interface GenerationOptions {
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  signal?: AbortSignal;
}

export interface ModelProviderMetadata {
  id: string;
  name: string;
  contextWindow: number;
  local: boolean;
}

export interface IModelProvider {
  metadata(): ModelProviderMetadata;
  generate(messages: AIMessage[], options?: GenerationOptions): Promise<string>;
  stream(messages: AIMessage[], options?: GenerationOptions): AsyncGenerator<AIStreamChunk>;
  healthCheck(): Promise<{ status: string; modelLoaded: boolean; model?: string }>;
}
