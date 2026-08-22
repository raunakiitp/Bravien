/** /api/conversations/[id] — read one conversation with its messages, update, delete. */

import { accountScope } from "@/lib/auth/guard";
import {
  deleteConversation,
  getConversation,
  getConversationMessages,
  updateConversation,
} from "@/lib/db/conversations";
import { badRequest, notFound } from "@/lib/security/errors";
import { z } from "zod";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Next 16 hands route params as a promise. */
type Params = { params: Promise<{ id: string }> };

const SUBJECT = "Conversation history";

export async function GET(_request: Request, { params }: Params) {
  const userId = await accountScope(SUBJECT);
  if (typeof userId !== "string") return userId;

  const { id } = await params;
  const conversation = await getConversation(userId, id);
  if (!conversation) return notFound("Conversation not found.");

  const messages = await getConversationMessages(userId, id);
  return Response.json(
    { conversation, messages: messages ?? [] },
    { headers: { "Cache-Control": "no-store" } },
  );
}

const patchSchema = z
  .object({
    title: z.string().min(1).max(200).optional(),
    pinned: z.boolean().optional(),
    archived: z.boolean().optional(),
    summary: z.string().max(4_000).nullable().optional(),
  })
  .strict()
  .refine((v) => Object.keys(v).length > 0, {
    message: "Provide at least one field to update.",
  });

export async function PATCH(request: Request, { params }: Params) {
  const userId = await accountScope(SUBJECT);
  if (typeof userId !== "string") return userId;

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return badRequest("Request body must be valid JSON.");
  }
  const parsed = patchSchema.safeParse(payload);
  if (!parsed.success) return badRequest("Invalid request.", parsed.error.issues);

  const { id } = await params;
  const updated = await updateConversation(userId, id, parsed.data);
  if (!updated) return notFound("Conversation not found.");
  return Response.json(updated);
}

export async function DELETE(_request: Request, { params }: Params) {
  const userId = await accountScope(SUBJECT);
  if (typeof userId !== "string") return userId;

  const { id } = await params;
  const deleted = await deleteConversation(userId, id);
  if (!deleted) return notFound("Conversation not found.");
  return new Response(null, { status: 204 });
}
