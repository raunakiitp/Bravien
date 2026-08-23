import { NextResponse } from "next/server";
import { z } from "zod";

import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { deleteMemory, getMemory, updateMemory } from "@/lib/memory/service";
import { logger } from "@/lib/observability/logger";
import {
  badRequest,
  jsonError,
  notFound,
  unauthorized,
} from "@/lib/security/errors";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const patchMemorySchema = z
  .object({
    content: z.string().min(1).max(4000).optional(),
    type: z
      .enum([
        "EXPLICIT",
        "PREFERENCE",
        "PROJECT",
        "CONVERSATION",
        "PROFILE",
        "INSTRUCTION",
        "FACT",
        "WORKFLOW",
      ])
      .optional(),
    projectId: z.string().max(64).optional().nullable(),
  })
  .strict();

export async function GET(
  _request: Request,
  props: { params: Promise<{ id: string }> },
) {
  const { id } = await props.params;
  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(503, "PERSISTENCE_UNAVAILABLE", "Database unavailable.");
  }

  const memory = await getMemory(userId, id);
  if (!memory) return notFound("Memory not found.");

  return NextResponse.json({ memory });
}

export async function PATCH(
  request: Request,
  props: { params: Promise<{ id: string }> },
) {
  const { id } = await props.params;
  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(503, "PERSISTENCE_UNAVAILABLE", "Database unavailable.");
  }

  let json: unknown;
  try {
    json = await request.json();
  } catch {
    return badRequest("Invalid JSON payload.");
  }

  const parsed = patchMemorySchema.safeParse(json);
  if (!parsed.success) {
    return badRequest("Invalid update payload.", parsed.error.issues);
  }

  if (parsed.data.projectId) {
    const { prisma } = await import("@/lib/db/prisma");
    const ownedProject = await prisma.project.findFirst({
      where: { id: parsed.data.projectId, userId },
      select: { id: true },
    });
    if (!ownedProject) {
      return badRequest("Specified project does not exist or belong to your account.");
    }
  }

  try {
    const updated = await updateMemory(userId, id, parsed.data);
    if (!updated) return notFound("Memory not found.");
    return NextResponse.json(updated);
  } catch (error) {
    logger.error("memory.patch_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "MEMORY_UPDATE_ERROR", "Could not update memory.");
  }
}

export async function DELETE(
  _request: Request,
  props: { params: Promise<{ id: string }> },
) {
  const { id } = await props.params;
  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(503, "PERSISTENCE_UNAVAILABLE", "Database unavailable.");
  }

  try {
    const success = await deleteMemory(userId, id);
    if (!success) return notFound("Memory not found.");
    return new NextResponse(null, { status: 204 });
  } catch (error) {
    logger.error("memory.delete_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "MEMORY_DELETE_ERROR", "Could not delete memory.");
  }
}
