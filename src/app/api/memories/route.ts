import { NextResponse } from "next/server";
import { z } from "zod";

import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import {
  clearMemories,
  createMemory,
  listMemories,
  searchMemories,
} from "@/lib/memory/service";
import { logger } from "@/lib/observability/logger";
import { badRequest, jsonError, unauthorized } from "@/lib/security/errors";
import type { MemoryType } from "@/types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const memoryTypeSchema = z.enum([
  "EXPLICIT",
  "PREFERENCE",
  "PROJECT",
  "CONVERSATION",
  "PROFILE",
  "INSTRUCTION",
  "FACT",
  "WORKFLOW",
]);

const createMemorySchema = z
  .object({
    content: z.string().min(1, "Memory content cannot be empty.").max(4000),
    type: memoryTypeSchema.default("EXPLICIT"),
    projectId: z.string().max(64).optional().nullable(),
    sourceConversationId: z.string().max(64).optional().nullable(),
  })
  .strict();

export async function GET(request: Request) {
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

  const { searchParams } = new URL(request.url);
  const q = searchParams.get("q");
  const typeParam = searchParams.get("type");
  const projectId = searchParams.get("projectId");

  try {
    if (q?.trim()) {
      const items = await searchMemories(userId, q.trim(), {
        projectId: projectId ?? undefined,
        limit: 50,
      });
      return NextResponse.json({ items });
    }

    const items = await listMemories(userId, {
      type: (typeParam as MemoryType) || undefined,
      projectId: projectId ?? undefined,
      take: 100,
    });
    return NextResponse.json({ items });
  } catch (error) {
    logger.error("memories.list_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "MEMORIES_ERROR", "Could not load memories.");
  }
}

export async function POST(request: Request) {
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

  const parsed = createMemorySchema.safeParse(json);
  if (!parsed.success) {
    return badRequest("Invalid memory input.", parsed.error.issues);
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
    const memory = await createMemory({
      userId,
      content: parsed.data.content,
      type: parsed.data.type,
      projectId: parsed.data.projectId,
      sourceConversationId: parsed.data.sourceConversationId,
    });
    return NextResponse.json(memory, { status: 201 });
  } catch (error) {
    logger.error("memories.create_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "MEMORY_CREATE_ERROR", "Could not save memory.");
  }
}

export async function DELETE(request: Request) {
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

  const { searchParams } = new URL(request.url);
  const projectId = searchParams.get("projectId");

  try {
    const count = await clearMemories(userId, {
      projectId: projectId ?? undefined,
    });
    return NextResponse.json({ cleared: count });
  } catch (error) {
    logger.error("memories.clear_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "MEMORIES_CLEAR_ERROR", "Could not clear memories.");
  }
}
