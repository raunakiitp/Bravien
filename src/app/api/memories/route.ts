/** /api/memories — the facts Bravien is allowed to carry between conversations. */

import { accountScope } from "@/lib/auth/guard";
import { isFeatureEnabled } from "@/lib/features";
import { createMemory, listMemories } from "@/lib/memory/service";
import { badRequest, forbidden } from "@/lib/security/errors";
import { z } from "zod";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_MEMORY_CHARS = 2_000;

async function guard(): Promise<string | Response> {
  // Checked ahead of the account: when the feature is off, whether you could have
  // signed in is beside the point.
  if (!isFeatureEnabled("memory")) {
    return forbidden("Memory is disabled (FEATURE_MEMORY).");
  }
  return accountScope("Memory");
}

export async function GET() {
  const userId = await guard();
  if (typeof userId !== "string") return userId;

  const items = await listMemories(userId);
  return Response.json({ items }, { headers: { "Cache-Control": "no-store" } });
}

const createSchema = z
  .object({
    content: z.string().trim().min(1).max(MAX_MEMORY_CHARS),
    type: z
      .enum(["EXPLICIT", "PREFERENCE", "PROJECT", "CONVERSATION"])
      .default("EXPLICIT"),
    sourceConversationId: z.string().max(64).nullable().optional(),
  })
  .strict();

export async function POST(request: Request) {
  const userId = await guard();
  if (typeof userId !== "string") return userId;

  let payload: unknown;
  try {
    payload = await request.json();
  } catch {
    return badRequest("Request body must be valid JSON.");
  }
  const parsed = createSchema.safeParse(payload);
  if (!parsed.success) return badRequest("Invalid request.", parsed.error.issues);

  const memory = await createMemory({
    userId,
    type: parsed.data.type,
    content: parsed.data.content,
    sourceConversationId: parsed.data.sourceConversationId ?? null,
  });
  return Response.json(memory, { status: 201 });
}
