import { NextResponse } from "next/server";
import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { executeTask } from "@/lib/tasks/executor";
import { badRequest, jsonError, unauthorized } from "@/lib/security/errors";
import { logger } from "@/lib/observability/logger";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface Params {
  params: Promise<{ id: string }>;
}

export async function POST(request: Request, { params }: Params) {
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

  let options: { timeoutMs?: number } = {};
  try {
    const body = await request.json();
    if (body && typeof body.timeoutMs === "number") {
      options.timeoutMs = Math.min(Math.max(body.timeoutMs, 1000), 60000);
    }
  } catch {
    // Optional body
  }

  try {
    const result = await executeTask(userId, id, options);
    return NextResponse.json({ result });
  } catch (error) {
    logger.error("tasks.execution_failed", {
      taskId: id,
      error: error instanceof Error ? error.message : String(error),
    });
    return badRequest(error instanceof Error ? error.message : "Task execution failed.");
  }
}
