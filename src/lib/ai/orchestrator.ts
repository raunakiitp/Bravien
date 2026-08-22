import { trimConversationHistory } from "@/lib/ai/context";
import { getDefaultModelId, getModel } from "@/lib/ai/models";
import { buildSystemPrompt } from "@/lib/ai/prompts";
import {
  getProviderForModel,
  ModelUnavailableError,
} from "@/lib/ai/providers";
import type { AIMessage, AIStreamChunk } from "@/types";

export interface StreamChatOptions {
  messages: AIMessage[];
  modelId?: string;
  tools?: StreamTextTools;
  signal?: AbortSignal;
  userPreferences?: string | null;
  memories?: string[];
  modelInstructions?: string | null;
  /** Approximate context budget before trimming older turns */
  maxContextTokens?: number;
}

type StreamTextTools = NonNullable<
  Parameters<
    ReturnType<typeof getProviderForModel>["streamText"]
  >[0]["tools"]
>;

export interface GenerateTitleOptions {
  messages: AIMessage[];
  modelId?: string;
  signal?: AbortSignal;
}

/**
 * Orchestrates a chat turn: system prompt, context trim, provider stream.
 * Yields SSE-friendly AIStreamChunk events.
 */
export async function* streamChat(
  options: StreamChatOptions,
): AsyncIterable<AIStreamChunk> {
  const modelId = options.modelId ?? getDefaultModelId();
  const model = getModel(modelId);

  if (!model) {
    yield {
      kind: "error",
      code: "MODEL_UNAVAILABLE",
      message: `Unknown model: ${modelId}`,
    };
    return;
  }

  let provider;
  try {
    provider = getProviderForModel(modelId);
  } catch (err) {
    if (err instanceof ModelUnavailableError) {
      yield { kind: "error", code: err.code, message: err.message };
      return;
    }
    throw err;
  }

  const system = buildSystemPrompt({
    userPreferences: options.userPreferences,
    memories: options.memories,
    modelInstructions: options.modelInstructions,
  });

  const withoutSystem = options.messages.filter((m) => m.role !== "system");
  const budget =
    options.maxContextTokens ??
    Math.max(4_000, Math.floor(model.contextWindow * 0.7));

  const trimmed = trimConversationHistory(
    [{ role: "system", content: system }, ...withoutSystem],
    budget,
  );

  try {
    yield* provider.streamText({
      model: model.providerModelId,
      messages: trimmed,
      tools: options.tools,
      signal: options.signal,
      maxTokens: model.maxOutputTokens,
    });
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Unexpected streaming failure";
    yield { kind: "error", code: "ORCHESTRATOR_ERROR", message };
  }
}

/**
 * Non-streaming title generation for a new conversation.
 */
export async function generateConversationTitle(
  options: GenerateTitleOptions,
): Promise<string> {
  const modelId = options.modelId ?? getDefaultModelId();
  const model = getModel(modelId);
  if (!model) return "New chat";

  const provider = getProviderForModel(modelId);
  if (!provider.completeText) {
    return fallbackTitle(options.messages);
  }

  const sample = options.messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .slice(0, 4)
    .map((m) => `${m.role}: ${m.content.slice(0, 400)}`)
    .join("\n");

  try {
    const title = await provider.completeText({
      model: model.providerModelId,
      messages: [
        {
          role: "system",
          content:
            "Generate a short conversation title (max 6 words). No quotes or punctuation fluff. Reply with the title only.",
        },
        {
          role: "user",
          content: `Conversation:\n${sample}`,
        },
      ],
      temperature: 0.4,
      maxTokens: 24,
      signal: options.signal,
    });
    const cleaned = title.replace(/^["']|["']$/g, "").trim();
    return cleaned.slice(0, 80) || fallbackTitle(options.messages);
  } catch {
    return fallbackTitle(options.messages);
  }
}

function fallbackTitle(messages: AIMessage[]): string {
  const firstUser = messages.find((m) => m.role === "user");
  if (!firstUser?.content.trim()) return "New chat";
  const line = firstUser.content.trim().split(/\n/)[0] ?? "New chat";
  return line.slice(0, 60);
}
