/** /api/memories/[id] — edit or forget a single memory. */

import { accountScope } from "@/lib/auth/guard";
import { isFeatureEnabled } from "@/lib/features";
import { deleteMemory, updateMemory } from "@/lib/memory/service";
import { badRequest, forbidden, notFound } from "@/lib/security/errors";
import { z } from "zod";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type Params = { params: Promise<{ id: string }> };

async function guard(): Promise<string | Response> {
  if (!isFeatureEnabled("memory")) {
    return forbidden("Memory is disabled (FEATURE_MEMORY).");
  }
  return accountScope("Memory");
}

const patchSchema = z
  .object({
    content: z.string().trim().min(1).max(2_000).optional(),
    type: z
      .enum(["EXPLICIT", "PREFERENCE", "PROJECT", "CONVERSATION"])
      .optional(),
  })
  .strict()
  .refine((v) => Object.keys(v).length > 0, {
    message: "Provide at least one field to update.",
  });

export async function PATCH(request: Request, { params }: Params) {
  const userId = await guard();
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
  const updated = await updateMemory(userId, id, parsed.data);
  if (!updated) return notFound("Memory not found.");
  return Response.json(updated);
}

export async function DELETE(_request: Request, { params }: Params) {
  const userId = await guard();
  if (typeof userId !== "string") return userId;

  const { id } = await params;
  const deleted = await deleteMemory(userId, id);
  if (!deleted) return notFound("Memory not found.");
  return new Response(null, { status: 204 });
}
