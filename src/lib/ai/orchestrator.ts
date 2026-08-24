/**
 * Chat orchestration: system prompt, context trimming, provider streaming.
 *
 * Everything yielded here originates from the local Bravien runtime. The only
 * chunks this module authors itself are `error` frames, which are labelled as
 * errors and never rendered as assistant text (§52).
 */

import { buildContext } from "@/lib/ai/context";
import { getDefaultModelId, getModel } from "@/lib/ai/models";
import { getProviderForModel, ModelUnavailableError } from "@/lib/ai/providers";
import type { AIMessage, AIStreamChunk, StreamTextParams } from "@/types";

type StreamTextTools = NonNullable<StreamTextParams["tools"]>;

export interface StreamChatOptions {
  messages: AIMessage[];
  modelId?: string;
  tools?: StreamTextTools;
  signal?: AbortSignal;
  userPreferences?: string | null;
  projectInstructions?: string | null;
  projectDocumentsContext?: string | null;
  agentStateSummary?: string | null;
  toolResultsFormatted?: string | null;
  evidenceFormatted?: string | null;
  planSummary?: string | null;
  conversationSummary?: string | null;
  isCodingMode?: boolean;
  memories?: string[];
  modelInstructions?: string | null;
  maxContextTokens?: number;
}

export interface GenerateTitleOptions {
  messages: AIMessage[];
  modelId?: string;
  signal?: AbortSignal;
}

/**
 * Orchestrates a chat turn: system prompt, context budgeting, tool injection, provider stream.
 * Yields SSE-friendly AIStreamChunk events.
 */
export async function* streamChat(
  options: StreamChatOptions,
): AsyncIterable<AIStreamChunk> {
  const modelId = options.modelId ?? (await getDefaultModelId());

  if (!modelId) {
    yield {
      kind: "error",
      code: "MODEL_UNAVAILABLE",
      message:
        "No Bravien checkpoint is loaded. Start the runtime with `python scripts/serve.py`.",
    };
    return;
  }

  const model = await getModel(modelId);
  let provider;
  try {
    provider = await getProviderForModel(modelId);
  } catch (err) {
    if (err instanceof ModelUnavailableError) {
      yield { kind: "error", code: err.code, message: err.message };
      return;
    }
    throw err;
  }

  if (!model) {
    yield {
      kind: "error",
      code: "MODEL_UNAVAILABLE",
      message: `Unknown model: ${modelId}`,
    };
    return;
  }

  const built = buildContext({
    messages: options.messages.filter((m) => m.role !== "system"),
    contextWindow: options.maxContextTokens ?? model.contextWindow,
    userPreferences: options.userPreferences,
    projectInstructions: options.projectInstructions,
    projectDocumentsContext: options.projectDocumentsContext,
    agentStateSummary: options.agentStateSummary,
    toolResultsFormatted: options.toolResultsFormatted,
    evidenceFormatted: options.evidenceFormatted,
    planSummary: options.planSummary,
    conversationSummary: options.conversationSummary,
    isCodingMode: options.isCodingMode,
    memories: options.memories,
  });

  const promptMessages: AIMessage[] = [
    { role: "system", content: built.systemPrompt },
    ...built.messages,
  ];

  try {
    yield* provider.streamText({
      model: model.providerModelId,
      messages: promptMessages,
      tools: options.tools,
      signal: options.signal,
      maxTokens: model.maxOutputTokens,
    });
  } catch (err) {
    if (isAbort(err, options.signal)) return;
    const message =
      err instanceof Error ? err.message : "Unexpected streaming failure";
    yield { kind: "error", code: "ORCHESTRATOR_ERROR", message };
  }
}

function isAbort(err: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true;
  return err instanceof Error && err.name === "AbortError";
}

/**
 * A title for a new conversation.
 *
 * Derived from the first user message by default. Asking the model to name the
 * conversation costs a second generation and, on a small checkpoint, produces
 * nonsense — so it is opt-in via `BRAVIEN_MODEL_TITLES=1` rather than on by
 * default and quietly bad. The derived title is plainly the user's own text, not
 * model output presented as such (§52).
 */
export async function generateConversationTitle(
  options: GenerateTitleOptions,
): Promise<string> {
  const useModel = process.env.BRAVIEN_MODEL_TITLES === "1";
  if (!useModel) return fallbackTitle(options.messages);

  const modelId = options.modelId ?? (await getDefaultModelId());
  if (!modelId) return fallbackTitle(options.messages);

  const model = await getModel(modelId);
  if (!model) return fallbackTitle(options.messages);

  let provider;
  try {
    provider = await getProviderForModel(modelId);
  } catch {
    return fallbackTitle(options.messages);
  }
  if (!provider.completeText) return fallbackTitle(options.messages);

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
        { role: "user", content: `Conversation:\n${sample}` },
      ],
      temperature: 0.4,
      maxTokens: 24,
      signal: options.signal,
    });
    const cleaned = title
      .replace(/^["']|["']$/g, "")
      .split(/\n/)[0]
      ?.trim();
    return cleaned?.slice(0, 80) || fallbackTitle(options.messages);
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
