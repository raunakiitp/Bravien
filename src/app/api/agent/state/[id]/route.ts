import { NextResponse } from "next/server";
import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import {
  getAgentStateById,
  updateAgentState,
  updateAgentStateSchema,
} from "@/lib/ai/agent-state";
import { badRequest, jsonError, unauthorized } from "@/lib/security/errors";
import { logger } from "@/lib/observability/logger";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface Params {
  params: Promise<{ id: string }>;
}

export async function GET(_request: Request, { params }: Params) {
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

  const { id } = await params;

  try {
    const state = await getAgentStateById(userId, id);
    return NextResponse.json({ state });
  } catch (error) {
    logger.warn("agent_state.get_failed", { id, error: String(error) });
    return jsonError(404, "NOT_FOUND", "Agent state not found or access denied.");
  }
}

export async function PATCH(request: Request, { params }: Params) {
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

  const { id } = await params;

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return badRequest("Invalid JSON body.");
  }

  const parsed = updateAgentStateSchema.safeParse(body);
  if (!parsed.success) {
    return badRequest(parsed.error.issues[0]?.message ?? "Invalid update payload.");
  }

  try {
    const updated = await updateAgentState(userId, id, parsed.data);
    return NextResponse.json({ state: updated });
  } catch (error) {
    logger.warn("agent_state.update_failed", { id, error: String(error) });
    return badRequest(error instanceof Error ? error.message : "Failed to update agent state.");
  }
}
