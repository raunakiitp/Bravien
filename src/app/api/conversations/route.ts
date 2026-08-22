/**
 * /api/conversations — list and create.
 *
 * Persistence requires a signed-in user and a reachable database. When either is
 * missing the route says so with a specific code rather than returning an empty
 * list that looks like "you have no conversations".
 */

import { getDefaultModelId } from "@/lib/ai/models";
import { accountScope } from "@/lib/auth/guard";
import { createConversation, listConversations } from "@/lib/db/conversations";
import { badRequest, jsonError } from "@/lib/security/errors";
import { z } from "zod";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const SUBJECT = "Conversation history";

const listQuerySchema = z.object({
  archived: z.enum(["true", "false"]).optional(),
  q: z.string().max(200).optional(),
  take: z.coerce.number().int().min(1).max(100).optional(),
  cursor: z.string().max(64).optional(),
});

export async function GET(request: Request) {
  const userId = await accountScope(SUBJECT);
  if (typeof userId !== "string") return userId;

  const url = new URL(request.url);
  const parsed = listQuerySchema.safeParse(Object.fromEntries(url.searchParams));
  if (!parsed.success) return badRequest("Invalid query.", parsed.error.issues);

  const { items, nextCursor } = await listConversations(userId, {
    archived: parsed.data.archived === "true",
    query: parsed.data.q,
    take: parsed.data.take,
    cursor: parsed.data.cursor,
  });

  return Response.json(
    { items, nextCursor },
    { headers: { "Cache-Control": "no-store" } },
  );
}

const createSchema = z
  .object({
    title: z.string().min(1).max(200).optional(),
    model: z.string().min(1).max(200).optional(),
  })
  .strict();

export async function POST(request: Request) {
  const userId = await accountScope(SUBJECT);
  if (typeof userId !== "string") return userId;

  let payload: unknown = {};
  try {
    const text = await request.text();
    if (text.trim()) payload = JSON.parse(text);
  } catch {
    return badRequest("Request body must be valid JSON.");
  }

  const parsed = createSchema.safeParse(payload);
  if (!parsed.success) return badRequest("Invalid request.", parsed.error.issues);

  const model = parsed.data.model ?? (await getDefaultModelId());
  if (!model) {
    return jsonError(
      503,
      "MODEL_UNAVAILABLE",
      "No Bravien checkpoint is loaded, so a conversation has no model to record.",
    );
  }

  const conversation = await createConversation({
    userId,
    title: parsed.data.title ?? "New chat",
    model,
  });
  return Response.json(conversation, { status: 201 });
}
