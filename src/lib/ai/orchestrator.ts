/**
 * Chat orchestration: system prompt, context trimming, provider streaming.
 *
 * Everything yielded here originates from the local Bravien runtime. The only
 * chunks this module authors itself are `error` frames, which are labelled as
 * errors and never rendered as assistant text (§52).
 */

import { boundHistoryForPayload } from "@/lib/ai/context";
import { getDefaultModelId, getModel } from "@/lib/ai/models";
import { buildSystemPromptDetailed } from "@/lib/ai/prompts";
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
  /**
   * Overrides the payload guard's allowance. Not a context budget — the runtime
   * owns that and reports it back on `message_complete`.
   */
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

  // The system prompt is sized to this model's real context window. The full
  // prompt is 817 tokens, which does not fit `bravien-tiny`'s 512-token window at
  // all, so a small checkpoint gets a smaller prompt rather than a truncated one.
  const { prompt: system, tier, droppedExtras } = buildSystemPromptDetailed({
    userPreferences: options.userPreferences,
    memories: options.memories,
    modelInstructions: options.modelInstructions,
    contextWindow: model.contextWindow,
  });

  if (droppedExtras.length > 0) {
    console.warn(
      `[bravien] ${model.id} (${model.contextWindow}-token context, "${tier}" prompt tier) ` +
        `has no room for: ${droppedExtras.join(", ")}`,
    );
  }

  const withoutSystem = options.messages.filter((m) => m.role !== "system");

  // A payload guard, not a context budget. The runtime owns the budget: it has
  // the tokenizer, it drops whole turns oldest-first, it keeps the system prompt,
  // and it reports the result on the `message_complete` frame. Trimming to a
  // character estimate here is what previously cut prompts to a third of their
  // size and took the system message with them.
  const bounded = boundHistoryForPayload(
    [{ role: "system", content: system }, ...withoutSystem],
    options.maxContextTokens ?? model.contextWindow,
  );

  try {
    yield* provider.streamText({
      model: model.providerModelId,
      messages: bounded,
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
