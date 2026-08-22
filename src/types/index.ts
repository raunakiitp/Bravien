/** Shared domain & AI types for Bravien */

export type AIModelCapability = "text" | "vision" | "tools" | "reasoning";

export interface AIModel {
  id: string;
  name: string;
  description: string;
  /** Upstream provider model identifier (e.g. gpt-4o-mini) */
  providerModelId: string;
  provider: "openai-compatible" | "anthropic";
  capabilities: AIModelCapability[];
  contextWindow: number;
  maxOutputTokens?: number;
  isDefault?: boolean;
}

export type AIMessageRole = "system" | "user" | "assistant" | "tool";

export interface AIToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface AIMessage {
  role: AIMessageRole;
  content: string;
  name?: string;
  toolCallId?: string;
  toolCalls?: AIToolCall[];
}

export type AIStreamChunk =
  | { kind: "content_delta"; delta: string }
  | {
      kind: "tool_start";
      toolCallId: string;
      name: string;
      arguments?: string;
    }
  | {
      kind: "tool_result";
      toolCallId: string;
      name: string;
      result: unknown;
    }
  | {
      kind: "citation";
      title: string;
      url: string;
      snippet?: string;
    }
  | {
      kind: "message_complete";
      finishReason?: string;
      usage?: { inputTokens?: number; outputTokens?: number };
    }
  | { kind: "error"; code: string; message: string };

export interface StreamTextParams {
  model: string;
  messages: AIMessage[];
  tools?: Array<{
    type: "function";
    function: {
      name: string;
      description: string;
      parameters: Record<string, unknown>;
    };
  }>;
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
}

export interface CompleteTextParams {
  model: string;
  messages: AIMessage[];
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
}

export interface AIProvider {
  id: string;
  streamText(params: StreamTextParams): AsyncIterable<AIStreamChunk>;
  completeText?(params: CompleteTextParams): Promise<string>;
}

export interface ConversationDTO {
  id: string;
  userId: string;
  title: string;
  model: string;
  pinned: boolean;
  archived: boolean;
  shareId: string | null;
  sharedAt: string | null;
  summary: string | null;
  metadata: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
}

export interface MessageDTO {
  id: string;
  conversationId: string;
  role: AIMessageRole;
  content: string;
  parentId: string | null;
  attachments: unknown;
  toolCalls: unknown;
  citations: unknown;
  feedback: "LIKE" | "DISLIKE" | null;
  metadata: Record<string, unknown> | null;
  createdAt: string;
}

export type MemoryType =
  | "EXPLICIT"
  | "PREFERENCE"
  | "PROJECT"
  | "CONVERSATION";

export interface MemoryDTO {
  id: string;
  userId: string;
  type: MemoryType;
  content: string;
  source: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface FeatureFlags {
  web_search: boolean;
  file_uploads: boolean;
  vision: boolean;
  voice: boolean;
  memory: boolean;
  image_generation: boolean;
  code_execution: boolean;
}

export interface ApiError {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}
