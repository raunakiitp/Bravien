import { NextResponse } from "next/server";
import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import {
  getOrCreateActiveAgentState,
  createAgentStateSchema,
} from "@/lib/ai/agent-state";
import { prisma } from "@/lib/db/prisma";
import { badRequest, jsonError, unauthorized } from "@/lib/security/errors";
import { logger } from "@/lib/observability/logger";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(request: Request) {
  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(503, "PERSISTENCE_UNAVAILABLE", "Database is unreachable.");
  }

  const { searchParams } = new URL(request.url);
  const projectId = searchParams.get("projectId") || undefined;
  const taskId = searchParams.get("taskId") || undefined;

  try {
    const state = await getOrCreateActiveAgentState(userId, {
      projectId,
      taskId,
    });
    return NextResponse.json({ state });
  } catch (error) {
    logger.error("agent_state.get_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "AGENT_STATE_ERROR", "Could not load agent state.");
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
    return jsonError(503, "PERSISTENCE_UNAVAILABLE", "Database is unreachable.");
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return badRequest("Invalid JSON body.");
  }

  const parsed = createAgentStateSchema.safeParse(body);
  if (!parsed.success) {
    return badRequest(parsed.error.issues[0]?.message ?? "Invalid state payload.");
  }

  try {
    const state = await getOrCreateActiveAgentState(userId, parsed.data);
    return NextResponse.json({ state }, { status: 201 });
  } catch (error) {
    logger.error("agent_state.create_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return badRequest(error instanceof Error ? error.message : "Failed to initialize agent state.");
  }
}
