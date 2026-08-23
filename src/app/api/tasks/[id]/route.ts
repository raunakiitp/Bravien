import { NextResponse } from "next/server";
import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { getTaskById, updateTask, deleteTask } from "@/lib/tasks/service";
import { updateTaskSchema } from "@/lib/tasks/types";
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
    const task = await getTaskById(userId, id);
    return NextResponse.json({ task });
  } catch (error) {
    logger.warn("tasks.get_failed", { taskId: id, error: String(error) });
    return jsonError(404, "TASK_NOT_FOUND", "Task not found or access denied.");
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

  const parsed = updateTaskSchema.safeParse(body);
  if (!parsed.success) {
    return badRequest(parsed.error.issues[0]?.message ?? "Invalid update payload.");
  }

  try {
    const updated = await updateTask(userId, id, parsed.data);
    return NextResponse.json({ task: updated });
  } catch (error) {
    logger.warn("tasks.update_failed", { taskId: id, error: String(error) });
    return badRequest(error instanceof Error ? error.message : "Failed to update task.");
  }
}

export async function DELETE(_request: Request, { params }: Params) {
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
    const res = await deleteTask(userId, id);
    return NextResponse.json(res);
  } catch (error) {
    logger.warn("tasks.delete_failed", { taskId: id, error: String(error) });
    return jsonError(404, "TASK_NOT_FOUND", "Task not found or access denied.");
  }
}
