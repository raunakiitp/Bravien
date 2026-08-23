import { NextResponse } from "next/server";
import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { listRecentActivities } from "@/lib/tasks/service";
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
  const limitParam = searchParams.get("limit");
  const limit = limitParam ? parseInt(limitParam, 10) : 30;

  try {
    const activities = await listRecentActivities(userId, {
      projectId,
      taskId,
      limit: isNaN(limit) ? 30 : limit,
    });
    return NextResponse.json({ items: activities });
  } catch (error) {
    logger.error("activity.list_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "ACTIVITY_ERROR", "Could not load activity log.");
  }
}
