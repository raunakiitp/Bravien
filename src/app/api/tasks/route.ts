import { NextResponse } from "next/server";
import { z } from "zod";
import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { createTask, listTasks } from "@/lib/tasks/service";
import { createTaskSchema, taskFilterSchema } from "@/lib/tasks/types";
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
  const parsed = taskFilterSchema.safeParse({
    projectId: searchParams.get("projectId") || undefined,
    status: searchParams.get("status") || undefined,
    type: searchParams.get("type") || undefined,
    search: searchParams.get("search") || undefined,
    limit: searchParams.get("limit") || undefined,
    offset: searchParams.get("offset") || undefined,
  });

  if (!parsed.success) {
    return badRequest("Invalid task filter query parameters.");
  }

  try {
    const result = await listTasks(userId, parsed.data);
    return NextResponse.json(result);
  } catch (error) {
    logger.error("tasks.list_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "TASKS_ERROR", "Could not load tasks.");
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

  const parsed = createTaskSchema.safeParse(body);
  if (!parsed.success) {
    return badRequest(parsed.error.issues[0]?.message ?? "Invalid task data.");
  }

  try {
    const task = await createTask(userId, parsed.data);
    return NextResponse.json({ task }, { status: 201 });
  } catch (error) {
    logger.error("tasks.create_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return badRequest(error instanceof Error ? error.message : "Failed to create task.");
  }
}
