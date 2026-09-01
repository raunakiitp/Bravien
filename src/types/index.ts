/** Shared domain & AI types for Bravien */

export type AIModelCapability = "text" | "vision" | "tools" | "reasoning";

/**
 * Bravien has exactly one inference backend: the local runtime serving a Bravien
 * checkpoint. No hosted model service may appear here — a Bravien response comes
 * from Bravien's own weights or it does not exist (§2, §76).
 */
export type AIProviderId = "bravien-local";

export interface AIModel {
  id: string;
  name: string;
  description: string;
  /** Model name as the local runtime reports it. */
  providerModelId: string;
  provider: AIProviderId;
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

export type AgentEventType =
  | "request_started"
  | "agent_started"
  | "routing"
  | "intent_detected"
  | "memory_loaded"
  | "memory_retrieved"
  | "memory_saved"
  | "state_loaded"
  | "planning"
  | "planning_started"
  | "plan_created"
  | "step_started"
  | "tool_selected"
  | "tool_started"
  | "tool_completed"
  | "research_started"
  | "retrieval_started"
  | "retrieval_completed"
  | "source_found"
  | "evidence_added"
  | "checkpoint_saved"
  | "confirmation_required"
  | "waiting_confirmation"
  | "step_completed"
  | "recovery_started"
  | "recovery_strategy_selected"
  | "recovery_completed"
  | "recovery_failed"
  | "correction_started"
  | "correction_completed"
  | "verification_started"
  | "verification_completed"
  | "response_started"
  | "agent_completed"
  | "agent_failed";

export type AIStreamChunk =
  | { kind: "content_delta"; delta: string }
  | {
      kind: "agent_event";
      eventType: AgentEventType;
      message: string;
      metadata?: Record<string, unknown>;
    }
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
      /**
       * The runtime's context accounting for this turn, measured with the real
       * tokenizer (§Phase 7). Present whenever the turn went through the chat
       * endpoint. `truncated` here is authoritative — the web side does not
       * compute it.
       */
      context?: {
        input_tokens: number;
        max_context_tokens: number;
        reserved_output_tokens: number;
        available_tokens: number;
        overhead_tokens: number;
        truncated: boolean;
        truncated_turns: number;
        system_truncated: boolean;
        latest_user_truncated: boolean;
      };
      /** Set only when a raw prompt had to be cut without message structure. */
      promptTruncation?: {
        truncated: boolean;
        prompt_tokens_before: number;
        prompt_tokens_after: number;
        dropped_tokens: number;
      };
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

/**
 * Frames sent by `/api/chat` over SSE.
 *
 * Model output travels only in `content_delta`. The extra kinds carry
 * bookkeeping — which conversation this became, whether it was saved — so the
 * client never has to infer state from the text, and text never has to carry
 * anything but what the model produced (§52).
 */
export type ChatStreamFrame =
  | AIStreamChunk
  | {
      kind: "meta";
      conversationId: string | null;
      model: string;
      persisted: boolean;
      title?: string;
    }
  | {
      kind: "saved";
      userMessageId: string | null;
      assistantMessageId: string | null;
    };


export interface ProjectDTO {
  id: string;
  userId: string;
  name: string;
  description: string | null;
  instructions: string | null;
  createdAt: string;
  updatedAt: string;
  conversationCount?: number;
  attachmentCount?: number;
}

export interface ConversationDTO {
  id: string;
  userId: string;
  projectId?: string | null;
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
  | "CONVERSATION"
  | "PROFILE"
  | "INSTRUCTION"
  | "FACT"
  | "WORKFLOW";

export interface MemoryDTO {
  id: string;
  userId: string;
  projectId?: string | null;
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
