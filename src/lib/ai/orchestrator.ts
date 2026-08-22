/**
 * Chat orchestration: system prompt, context trimming, provider streaming.
 *
 * Everything yielded here originates from the local Bravien runtime. The only
 * chunks this module authors itself are `error` frames, which are labelled as
 * errors and never rendered as assistant text (§52).
 */

import { trimConversationHistory } from "@/lib/ai/context";
import { getDefaultModelId, getModel } from "@/lib/ai/models";
import { buildSystemPrompt } from "@/lib/ai/prompts";
import { getProviderForModel, ModelUnavailableError } from "@/lib/ai/providers";
import type { AIMessage, AIStreamChunk, StreamTextParams } from "@/types";

type StreamTextTools = NonNullable<StreamTextParams["tools"]>;

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
    // getProviderForModel throws when the model is unknown, so this is
    // unreachable; it keeps the type narrow without a non-null assertion.
    yield {
      kind: "error",
      code: "MODEL_UNAVAILABLE",
      message: `Unknown model: ${modelId}`,
    };
    return;
  }

  const system = buildSystemPrompt({
    userPreferences: options.userPreferences,
    memories: options.memories,
    modelInstructions: options.modelInstructions,
  });

  const withoutSystem = options.messages.filter((m) => m.role !== "system");

  // Reserve room for the answer inside the model's real context window. Small
  // checkpoints have small windows, so the old 4k floor would overflow them.
  const budget =
    options.maxContextTokens ?? Math.floor(model.contextWindow * 0.6);

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
