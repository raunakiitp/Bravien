/**
 * POST /api/chat — stream a Bravien turn.
 *
 * The only source of assistant text in this file is the byte stream coming back
 * from the local runtime. Nothing here writes, pads, paces or re-chunks model
 * output; deltas are forwarded as they arrive (§52). If the runtime cannot
 * answer, the response is an `error` frame that the UI renders as an error — not
 * as something Bravien said.
 */

import { getDefaultModelId } from "@/lib/ai/models";
import {
  generateConversationTitle,
  streamChat,
} from "@/lib/ai/orchestrator";
import { getCurrentUser } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import {
  appendMessage,
  createConversation,
  getConversation,
  touchConversation,
} from "@/lib/db/conversations";
import { isFeatureEnabled } from "@/lib/features";
import { getMemoryContentsForPrompt } from "@/lib/memory/service";
import { logger } from "@/lib/observability/logger";
import { badRequest, jsonError, tooManyRequests } from "@/lib/security/errors";
import { rateLimit } from "@/lib/security/rate-limit";
import type { AIMessage, ChatStreamFrame } from "@/types";
import { z } from "zod";

export const dynamic = "force-dynamic";
/** Node runtime: the local runtime is on loopback and persistence needs Prisma. */
export const runtime = "nodejs";

/** Ceilings that bound work before any model is touched. */
const MAX_MESSAGES = 200;
const MAX_MESSAGE_CHARS = 100_000;
const MAX_TOTAL_CHARS = 400_000;
const MAX_BODY_BYTES = 1 << 20;

const RATE_LIMIT = 30;
const RATE_WINDOW_MS = 60_000;

const messageSchema = z
  .object({
    role: z.enum(["system", "user", "assistant"]),
    content: z.string().max(MAX_MESSAGE_CHARS),
  })
  .strict();

const bodySchema = z
  .object({
    messages: z.array(messageSchema).min(1).max(MAX_MESSAGES),
    modelId: z.string().min(1).max(200).optional(),
    conversationId: z.string().min(1).max(64).optional(),
    projectId: z.string().min(1).max(64).optional().nullable(),
    /** Client-side id for the user message, echoed back so the UI can reconcile. */
    clientMessageId: z.string().min(1).max(64).optional(),
    /** Regenerating: do not persist the user message again. */
    regenerate: z.boolean().optional(),
  })
  .strict();

function frame(data: ChatStreamFrame): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

/** Identify a caller for rate limiting. A user id when known, else the peer. */
function rateLimitKey(request: Request, userId: string | null): string {
  if (userId) return `chat:user:${userId}`;
  const forwarded = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  return `chat:ip:${forwarded || "local"}`;
}

/**
 * The user's own instructions and stored memories.
 *
 * Best-effort: a failure here degrades personalisation, and losing the turn over
 * it would be worse than answering without it.
 */
async function loadPersonalisation(
  userId: string,
  projectId?: string | null,
  query?: string,
): Promise<{
  instructions: string | null;
  projectInstructions: string | null;
  memories: string[];
}> {
  try {
    const { prisma } = await import("@/lib/db/prisma");
    const [user, project, memories] = await Promise.all([
      prisma.user.findUnique({
        where: { id: userId },
        select: {
          personalInstructions: true,
          responseStyle: true,
          memoryEnabled: true,
        },
      }),
      projectId
        ? prisma.project.findFirst({
            where: { id: projectId, userId },
            select: { instructions: true },
          })
        : Promise.resolve(null),
      isFeatureEnabled("memory")
        ? getMemoryContentsForPrompt(userId, { projectId, query, take: 10 })
        : Promise.resolve([]),
    ]);

    const parts = [user?.personalInstructions, user?.responseStyle]
      .map((p) => p?.trim())
      .filter((p): p is string => Boolean(p));

    return {
      instructions: parts.length ? parts.join("\n\n") : null,
      projectInstructions: project?.instructions?.trim() || null,
      memories: user?.memoryEnabled === false ? [] : memories,
    };
  } catch {
    return { instructions: null, projectInstructions: null, memories: [] };
  }
}

export async function POST(request: Request) {
  // Reject an oversized body from the header before reading it into memory. A
  // chunked request without content-length is still bounded by the byte count
  // check below.
  const declared = Number(request.headers.get("content-length") ?? 0);
  if (Number.isFinite(declared) && declared > MAX_BODY_BYTES) {
    return jsonError(413, "PAYLOAD_TOO_LARGE", "Request body is too large.");
  }

  let raw: string;
  try {
    raw = await request.text();
  } catch {
    return badRequest("Could not read the request body.");
  }
  if (Buffer.byteLength(raw, "utf8") > MAX_BODY_BYTES) {
    return jsonError(413, "PAYLOAD_TOO_LARGE", "Request body is too large.");
  }

  let parsedJson: unknown;
  try {
    parsedJson = JSON.parse(raw);
  } catch {
    return badRequest("Request body must be valid JSON.");
  }

  const parsed = bodySchema.safeParse(parsedJson);
  if (!parsed.success) {
    return badRequest("Invalid request.", parsed.error.issues);
  }
  const body = parsed.data;

  const totalChars = body.messages.reduce((n, m) => n + m.content.length, 0);
  if (totalChars > MAX_TOTAL_CHARS) {
    return badRequest(
      `Conversation is too long (${totalChars} characters, limit ${MAX_TOTAL_CHARS}).`,
    );
  }
  if (!body.messages.some((m) => m.role === "user" && m.content.trim())) {
    return badRequest("At least one non-empty user message is required.");
  }

  const user = await getCurrentUser().catch(() => null);
  const userId = user?.id ?? null;

  const limit = rateLimit(rateLimitKey(request, userId), RATE_LIMIT, RATE_WINDOW_MS);
  if (!limit.allowed) {
    return tooManyRequests(
      `Too many requests. Try again in ${Math.ceil((limit.resetAt - Date.now()) / 1000)}s.`,
    );
  }

  const modelId = body.modelId ?? (await getDefaultModelId());
  if (!modelId) {
    return jsonError(
      503,
      "MODEL_UNAVAILABLE",
      "No Bravien checkpoint is loaded. Start the runtime with `python scripts/serve.py`.",
    );
  }

  // Persistence is opt-in on configuration, not required. Without a database (or
  // without a signed-in user) the turn still runs; it just is not saved, and the
  // `meta` frame says so.
  const canPersist = Boolean(userId) && (await isDatabaseReachable());

  const messages: AIMessage[] = body.messages.map((m) => ({
    role: m.role,
    content: m.content,
  }));
  const lastUser = [...messages].reverse().find((m) => m.role === "user");

  let conversationId: string | null = body.conversationId ?? null;
  let activeProjectId: string | null = body.projectId ?? null;
  let conversationTitle: string | undefined;
  let userMessageId: string | null = null;

  if (canPersist && userId) {
    try {
      if (conversationId) {
        const existing = await getConversation(userId, conversationId);
        if (!existing) {
          // An unknown or foreign id must not silently become a new
          // conversation — that would hide a bug or an access attempt.
          return jsonError(404, "NOT_FOUND", "Conversation not found.");
        }
        if (existing.projectId) {
          activeProjectId = existing.projectId;
        }
      } else {
        conversationTitle = await generateConversationTitle({ messages });
        const created = await createConversation({
          userId,
          projectId: activeProjectId,
          title: conversationTitle,
          model: modelId,
        });
        conversationId = created.id;
      }

      if (!body.regenerate && lastUser) {
        const saved = await appendMessage({
          conversationId,
          role: "user",
          content: lastUser.content,
          metadata: body.clientMessageId
            ? { clientMessageId: body.clientMessageId }
            : null,
        });
        userMessageId = saved.id;
      }
    } catch (error) {
      logger.error("chat.persist_failed", {
        error: error instanceof Error ? error.message : String(error),
      });
      return jsonError(
        500,
        "PERSISTENCE_ERROR",
        "Could not save the conversation. The message was not sent.",
      );
    }
  }

  const abort = new AbortController();
  // A client that closes the tab should stop the generation, not leave the
  // runtime producing tokens nobody will read.
  request.signal.addEventListener("abort", () => abort.abort(), { once: true });

  // Personalisation & project instructions (with query-based relevance scoring)
  const personalisation = canPersist && userId
    ? await loadPersonalisation(userId, activeProjectId, lastUser?.content)
    : { instructions: null, projectInstructions: null, memories: [] as string[] };

  const encoder = new TextEncoder();
  const persistedConversationId = conversationId;
  const persistTarget = canPersist ? persistedConversationId : null;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (data: ChatStreamFrame) => {
        controller.enqueue(encoder.encode(frame(data)));
      };

      let assistantText = "";
      let failed = false;
      let sawComplete = false;

      try {
        send({
          kind: "meta",
          conversationId: persistedConversationId,
          model: modelId,
          persisted: canPersist,
          ...(conversationTitle ? { title: conversationTitle } : {}),
        });

        const { runUnifiedAgentTurn } = await import("@/lib/ai/agent-orchestrator");

        for await (const chunk of runUnifiedAgentTurn({
          userId: userId ?? undefined,
          projectId: activeProjectId,
          conversationId,
          messages,
          modelId,
          signal: abort.signal,
          userPreferences: personalisation.instructions,
          projectInstructions: personalisation.projectInstructions,
          memories: personalisation.memories,
        })) {
          if (chunk.kind === "content_delta") assistantText += chunk.delta;
          if (chunk.kind === "error") failed = true;
          if (chunk.kind === "message_complete") sawComplete = true;
          send(chunk);
        }

        if (!failed && !sawComplete && assistantText.length > 0) {
          send({
            kind: "message_complete",
            finishReason: "stop",
            usage: {
              outputTokens: Math.ceil(assistantText.length / 4),
            },
          });
        }
      } catch (error) {
        failed = true;
        const message =
          error instanceof Error ? error.message : "Streaming failed.";
        logger.error("chat.stream_failed", { error: message });
        send({ kind: "error", code: "STREAM_FAILED", message });
      }

      // Save whatever the model actually produced. A partial answer is still the
      // model's output and is worth keeping; an empty one is not a message.
      let assistantMessageId: string | null = null;
      if (persistTarget && assistantText.length > 0) {
        try {
          const saved = await appendMessage({
            conversationId: persistTarget,
            role: "assistant",
            content: assistantText,
            parentId: userMessageId,
            metadata: { model: modelId, ...(failed ? { incomplete: true } : {}) },
          });
          assistantMessageId = saved.id;
          await touchConversation(persistTarget);
        } catch (error) {
          logger.error("chat.save_reply_failed", {
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }

      send({ kind: "saved", userMessageId, assistantMessageId });
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },

    cancel() {
      abort.abort();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      // Proxies that buffer would defeat streaming entirely.
      "X-Accel-Buffering": "no",
    },
  });
}
