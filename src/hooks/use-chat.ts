"use client";

/**
 * The chat loop.
 *
 * Assistant text in this file has exactly one source: the `content_delta` frames
 * arriving from `/api/chat`. Nothing is typed out on a timer, padded, smoothed or
 * synthesised, and an `error` frame becomes an error on the turn rather than
 * words attributed to the model (§52). A turn that produced nothing shows as a
 * failure, not as silence from Bravien.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { AIMessageRole, ChatStreamFrame, MessageDTO } from "@/types";
import { apiErrorFrom } from "@/types/api";

/** A document whose extracted text travels with the message. */
export interface ChatAttachment {
  name: string;
  text: string;
}

export interface ChatSendOptions {
  attachments?: ChatAttachment[];
}

export type TurnStatus = "pending" | "streaming" | "done" | "failed" | "stopped";

export interface TurnError {
  code: string;
  message: string;
}

export interface UiMessage {
  /** Stable client id. Replaced by the database id once the turn is saved. */
  id: string;
  role: Extract<AIMessageRole, "user" | "assistant">;
  /** For assistant turns: only ever the concatenation of `content_delta`. */
  content: string;
  status: TurnStatus;
  error: TurnError | null;
  /** Which checkpoint produced this, as the server reported it. */
  model?: string;
  finishReason?: string;
  usage?: { inputTokens?: number; outputTokens?: number };
  /**
   * The runtime's context accounting for this turn, measured with the real
   * tokenizer. Shown so a truncated conversation is visible to the user rather
   * than only present in a server log (§Phase 7).
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
  /** Text pulled out of attached documents, sent with the message but shown apart. */
  attachments?: Array<{ name: string; characters: number }>;
  createdAt: number;
}

export interface UseChatOptions {
  conversationId?: string | null;
  initialMessages?: MessageDTO[];
  modelId?: string | null;
  /** Called with the id the server assigned when a new conversation is saved. */
  onConversationCreated?: (id: string, title?: string) => void;
  onPersistedChange?: (persisted: boolean) => void;
}

let counter = 0;
/** Ids only need to be unique within this tab; `crypto.randomUUID` needs HTTPS. */
function localId(prefix: string): string {
  counter += 1;
  return `${prefix}-${counter}-${Math.random().toString(36).slice(2, 8)}`;
}

function fromDto(dto: MessageDTO): UiMessage | null {
  if (dto.role !== "user" && dto.role !== "assistant") return null;
  const metadata = (dto.metadata ?? {}) as {
    model?: unknown;
    incomplete?: unknown;
  };
  return {
    id: dto.id,
    role: dto.role,
    content: dto.content,
    status: metadata.incomplete === true ? "stopped" : "done",
    error: null,
    model: typeof metadata.model === "string" ? metadata.model : undefined,
    createdAt: new Date(dto.createdAt).getTime(),
  };
}

/**
 * Split an SSE byte stream into `data:` payloads.
 *
 * Buffered across reads: the transport can cut a frame mid-JSON, and parsing
 * per-chunk would silently drop those deltas.
 */
async function* readFrames(
  body: ReadableStream<Uint8Array>,
  signal: AbortSignal,
): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (!signal.aborted) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const line of block.split("\n")) {
          if (line.startsWith("data: ")) yield line.slice(6);
        }
        boundary = buffer.indexOf("\n\n");
      }
    }
    const tail = buffer.trim();
    if (tail.startsWith("data: ")) yield tail.slice(6);
  } finally {
    reader.releaseLock();
  }
}

export function useChat(options: UseChatOptions = {}) {
  const [messages, setMessages] = useState<UiMessage[]>(() =>
    (options.initialMessages ?? [])
      .map(fromDto)
      .filter((m): m is UiMessage => m !== null),
  );
  const [conversationId, setConversationId] = useState<string | null>(
    options.conversationId ?? null,
  );
  const [busy, setBusy] = useState(false);
  const [persisted, setPersisted] = useState<boolean | null>(null);
  const [model, setModel] = useState<string | null>(options.modelId ?? null);

  const abortRef = useRef<AbortController | null>(null);

  // Latest-value refs, synced after commit rather than written during render.
  // Every reader is an event handler or async continuation, so "the last
  // committed value" is exactly what they want: `send` needs the turns that
  // existed *before* the one being added.
  const messagesRef = useRef(messages);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  });

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const patch = useCallback((id: string, change: Partial<UiMessage>) => {
    setMessages((current) =>
      current.map((m) => (m.id === id ? { ...m, ...change } : m)),
    );
  }, []);

  /**
   * Run one turn against `/api/chat`.
   *
   * `history` is the conversation as the model should see it; `placeholderId` is
   * the empty assistant turn already on screen that deltas append to.
   */
  const run = useCallback(
    async (
      history: UiMessage[],
      placeholderId: string,
      opts: { regenerate?: boolean; clientMessageId?: string } = {},
    ) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setBusy(true);

      let received = "";
      let sawComplete = false;
      let failure: TurnError | null = null;

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            messages: history.map((m) => ({
              role: m.role,
              content: m.content,
            })),
            ...(optionsRef.current.modelId
              ? { modelId: optionsRef.current.modelId }
              : {}),
            ...(conversationId ? { conversationId } : {}),
            ...(opts.clientMessageId
              ? { clientMessageId: opts.clientMessageId }
              : {}),
            ...(opts.regenerate ? { regenerate: true } : {}),
          }),
        });

        if (!response.ok) {
          const error = await apiErrorFrom(response);
          patch(placeholderId, {
            status: "failed",
            error: { code: error.code, message: error.message },
          });
          return;
        }
        if (!response.body) {
          patch(placeholderId, {
            status: "failed",
            error: {
              code: "NO_STREAM",
              message: "The server accepted the request but sent no stream.",
            },
          });
          return;
        }

        for await (const payload of readFrames(response.body, controller.signal)) {
          if (payload === "[DONE]") break;

          let frame: ChatStreamFrame;
          try {
            frame = JSON.parse(payload) as ChatStreamFrame;
          } catch {
            // A malformed frame is a transport bug. Dropping it is right: it is
            // not model output and must not be rendered as if it were.
            continue;
          }

          switch (frame.kind) {
            case "meta": {
              setModel(frame.model);
              setPersisted(frame.persisted);
              optionsRef.current.onPersistedChange?.(frame.persisted);
              patch(placeholderId, { model: frame.model, status: "streaming" });
              if (frame.conversationId && frame.conversationId !== conversationId) {
                setConversationId(frame.conversationId);
                optionsRef.current.onConversationCreated?.(
                  frame.conversationId,
                  frame.title,
                );
              }
              break;
            }
            case "content_delta": {
              received += frame.delta;
              patch(placeholderId, { content: received, status: "streaming" });
              break;
            }
            case "message_complete": {
              sawComplete = true;
              patch(placeholderId, {
                status: "done",
                finishReason: frame.finishReason,
                usage: frame.usage,
                context: frame.context,
              });
              break;
            }
            case "error": {
              failure = { code: frame.code, message: frame.message };
              break;
            }
            case "saved": {
              // Adopt the database ids so later edits and feedback address rows
              // that exist.
              const assistantId = frame.assistantMessageId;
              if (assistantId) {
                setMessages((current) =>
                  current.map((m) =>
                    m.id === placeholderId ? { ...m, id: assistantId } : m,
                  ),
                );
              }
              if (frame.userMessageId && opts.clientMessageId) {
                setMessages((current) =>
                  current.map((m) =>
                    m.id === opts.clientMessageId
                      ? { ...m, id: frame.userMessageId! }
                      : m,
                  ),
                );
              }
              break;
            }
            default:
              // tool_start / tool_result / citation: no tool-using checkpoint
              // exists yet, so there is nothing honest to render for these.
              break;
          }
        }
      } catch (cause) {
        if (controller.signal.aborted) {
          // A deliberate stop. The partial text stays: the model did produce it,
          // and a `saved` frame never arrived to rename the placeholder.
          patch(placeholderId, { status: "stopped" });
          return;
        }
        failure = {
          code: "NETWORK",
          message:
            cause instanceof Error
              ? cause.message
              : "The connection to Bravien dropped.",
        };
      } finally {
        abortRef.current = null;
        setBusy(false);
      }

      if (failure) {
        patch(placeholderId, {
          status: "failed",
          error: failure,
        });
      } else if (received.length === 0) {
        patch(placeholderId, {
          status: "failed",
          error: {
            code: "EMPTY_RESPONSE",
            message: "The runtime returned no tokens for this turn.",
          },
        });
      } else if (!sawComplete) {
        patch(placeholderId, { status: "stopped" });
      }
    },
    [conversationId, patch],
  );

  const send = useCallback(
    async (text: string, sendOptions: ChatSendOptions = {}) => {
      const body = text.trim();
      if (!body || abortRef.current) return;

      // Document text is prepended to what the model reads but kept out of the
      // bubble, so the user sees their message and not a wall of file contents.
      const documents = sendOptions.attachments ?? [];
      const prompt = documents.length
        ? `${documents
            .map(
              (doc) =>
                `<document name="${doc.name}">\n${doc.text}\n</document>`,
            )
            .join("\n\n")}\n\n${body}`
        : body;

      const userId = localId("user");
      const userMessage: UiMessage = {
        id: userId,
        role: "user",
        content: body,
        status: "done",
        error: null,
        attachments: documents.length
          ? documents.map((d) => ({ name: d.name, characters: d.text.length }))
          : undefined,
        createdAt: Date.now(),
      };

      const placeholderId = localId("assistant");
      const placeholder: UiMessage = {
        id: placeholderId,
        role: "assistant",
        content: "",
        status: "pending",
        error: null,
        createdAt: Date.now(),
      };

      setMessages((current) => [...current, userMessage, placeholder]);

      // What the model sees: prior turns plus the prompt, attachments included.
      const history = [
        ...messagesRef.current.filter((m) => m.status !== "failed"),
        { ...userMessage, content: prompt },
      ];
      await run(history, placeholderId, { clientMessageId: userId });
    },
    [run],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  /** Re-run the last turn. The previous answer is dropped, not kept alongside. */
  const regenerate = useCallback(async () => {
    if (abortRef.current) return;
    const current = messagesRef.current;
    const lastAssistant = [...current]
      .reverse()
      .findIndex((m) => m.role === "assistant");
    if (lastAssistant === -1) return;

    const cutAt = current.length - 1 - lastAssistant;
    const history = current.slice(0, cutAt);
    if (!history.some((m) => m.role === "user")) return;

    const placeholderId = localId("assistant");
    setMessages([
      ...history,
      {
        id: placeholderId,
        role: "assistant",
        content: "",
        status: "pending",
        error: null,
        createdAt: Date.now(),
      },
    ]);
    await run(history, placeholderId, { regenerate: true });
  }, [run]);

  /** Replace a user message and re-answer from that point. */
  const editAndResend = useCallback(
    async (messageId: string, text: string) => {
      const body = text.trim();
      if (!body || abortRef.current) return;

      const current = messagesRef.current;
      const index = current.findIndex((m) => m.id === messageId);
      if (index === -1 || current[index].role !== "user") return;

      const edited: UiMessage = {
        ...current[index],
        content: body,
        createdAt: Date.now(),
      };
      const history = [...current.slice(0, index), edited];

      const placeholderId = localId("assistant");
      setMessages([
        ...history,
        {
          id: placeholderId,
          role: "assistant",
          content: "",
          status: "pending",
          error: null,
          createdAt: Date.now(),
        },
      ]);
      await run(history, placeholderId, { regenerate: true });
    },
    [run],
  );

  const reset = useCallback(
    (next?: { conversationId?: string | null; messages?: MessageDTO[] }) => {
      abortRef.current?.abort();
      abortRef.current = null;
      setBusy(false);
      setConversationId(next?.conversationId ?? null);
      setMessages(
        (next?.messages ?? [])
          .map(fromDto)
          .filter((m): m is UiMessage => m !== null),
      );
    },
    [],
  );

  return {
    messages,
    busy,
    conversationId,
    persisted,
    model,
    send,
    stop,
    regenerate,
    editAndResend,
    reset,
  };
}
