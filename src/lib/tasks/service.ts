import { prisma } from "@/lib/db/prisma";
import { logger } from "@/lib/observability/logger";
import type { Prisma, TaskStatus, TaskPriority, TaskType } from "@prisma/client";
import {
  type CreateTaskInput,
  type UpdateTaskInput,
  type TaskFilterInput,
  createTaskSchema,
  updateTaskSchema,
  taskFilterSchema,
} from "./types";

/**
 * Strips potential sensitive fields from metadata before logging.
 */
function sanitizeMetadata(metadata?: Record<string, unknown> | null): Record<string, unknown> | undefined {
  if (!metadata) return undefined;
  const safe: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (/password|secret|key|token|auth|credential/i.test(key)) {
      safe[key] = "[REDACTED]";
    } else if (typeof value === "object" && value !== null) {
      safe[key] = sanitizeMetadata(value as Record<string, unknown>);
    } else {
      safe[key] = value;
    }
  }
  return safe;
}

/**
 * Creates an activity log entry with sanitized metadata.
 */
export async function logActivity(
  userId: string,
  data: {
    eventType: string;
    description: string;
    projectId?: string | null;
    taskId?: string | null;
    metadata?: Record<string, unknown>;
  },
) {
  try {
    return await prisma.activityLog.create({
      data: {
        userId,
        eventType: data.eventType,
        description: data.description,
        projectId: data.projectId ?? null,
        taskId: data.taskId ?? null,
        metadata: (sanitizeMetadata(data.metadata) as Prisma.InputJsonValue) ?? undefined,
      },
    });
  } catch (err) {
    logger.warn("activity.log_failed", {
      error: err instanceof Error ? err.message : String(err),
    });
    return null;
  }
}

/**
 * Retrieves recent activity logs for a user.
 */
export async function listRecentActivities(
  userId: string,
  options: { projectId?: string | null; taskId?: string | null; limit?: number } = {},
) {
  const limit = Math.min(options.limit ?? 20, 100);
  const where: Prisma.ActivityLogWhereInput = {
    userId,
    ...(options.projectId ? { projectId: options.projectId } : {}),
    ...(options.taskId ? { taskId: options.taskId } : {}),
  };

  return prisma.activityLog.findMany({
    where,
    orderBy: { createdAt: "desc" },
    take: limit,
    include: {
      project: { select: { id: true, name: true } },
      task: { select: { id: true, title: true, status: true } },
    },
  });
}

/**
 * Creates a new task for a user with project verification.
 */
export async function createTask(userId: string, rawInput: CreateTaskInput) {
  const input = createTaskSchema.parse(rawInput);

  if (input.projectId) {
    const project = await prisma.project.findFirst({
      where: { id: input.projectId, userId },
      select: { id: true },
    });
    if (!project) {
      throw new Error("Project not found or unauthorized access.");
    }
  }

  const task = await prisma.task.create({
    data: {
      userId,
      projectId: input.projectId ?? null,
      title: input.title,
      description: input.description ?? null,
      type: input.type as TaskType,
      priority: input.priority as TaskPriority,
      status: "PENDING",
      metadata: (sanitizeMetadata(input.metadata) as Prisma.InputJsonValue) ?? undefined,
    },
    include: {
      project: { select: { id: true, name: true } },
    },
  });

  await logActivity(userId, {
    eventType: "TASK_CREATED",
    description: `Created task "${task.title}" (${task.type})`,
    projectId: task.projectId,
    taskId: task.id,
  });

  return task;
}

/**
 * Fetches a task by ID with ownership enforcement.
 */
export async function getTaskById(userId: string, taskId: string) {
  const task = await prisma.task.findFirst({
    where: { id: taskId, userId },
    include: {
      project: { select: { id: true, name: true, instructions: true } },
      executions: {
        orderBy: { startedAt: "desc" },
        take: 10,
      },
      steps: {
        orderBy: { stepNumber: "asc" },
      },
    },
  });

  if (!task) {
    throw new Error("Task not found or access denied.");
  }

  return task;
}

/**
 * Lists tasks for a user with flexible filtering.
 */
export async function listTasks(userId: string, rawFilter: TaskFilterInput = {}) {
  const filter = taskFilterSchema.parse(rawFilter);

  const where: Prisma.TaskWhereInput = {
    userId,
    ...(filter.projectId ? { projectId: filter.projectId } : {}),
    ...(filter.status ? { status: filter.status as TaskStatus } : {}),
    ...(filter.type ? { type: filter.type as TaskType } : {}),
    ...(filter.search
      ? {
          OR: [
            { title: { contains: filter.search, mode: "insensitive" } },
            { description: { contains: filter.search, mode: "insensitive" } },
          ],
        }
      : {}),
  };

  const [total, tasks] = await Promise.all([
    prisma.task.count({ where }),
    prisma.task.findMany({
      where,
      orderBy: [{ priority: "desc" }, { updatedAt: "desc" }],
      take: filter.limit,
      skip: filter.offset,
      include: {
        project: { select: { id: true, name: true } },
        _count: { select: { steps: true, executions: true } },
      },
    }),
  ]);

  return { total, tasks, limit: filter.limit, offset: filter.offset };
}

/**
 * Updates a task with ownership verification.
 */
export async function updateTask(userId: string, taskId: string, rawInput: UpdateTaskInput) {
  const input = updateTaskSchema.parse(rawInput);

  // Check ownership
  const existing = await prisma.task.findFirst({
    where: { id: taskId, userId },
    select: { id: true, status: true, title: true, projectId: true },
  });

  if (!existing) {
    throw new Error("Task not found or access denied.");
  }

  const updated = await prisma.task.update({
    where: { id: taskId },
    data: {
      ...(input.title !== undefined ? { title: input.title } : {}),
      ...(input.description !== undefined ? { description: input.description } : {}),
      ...(input.type !== undefined ? { type: input.type as TaskType } : {}),
      ...(input.status !== undefined ? { status: input.status as TaskStatus } : {}),
      ...(input.priority !== undefined ? { priority: input.priority as TaskPriority } : {}),
      ...(input.resultSummary !== undefined ? { resultSummary: input.resultSummary } : {}),
      ...(input.status === "COMPLETED" ? { completedAt: new Date() } : {}),
      ...(input.metadata !== undefined
        ? { metadata: sanitizeMetadata(input.metadata) as Prisma.InputJsonValue }
        : {}),
    },
    include: {
      project: { select: { id: true, name: true } },
    },
  });

  if (input.status && input.status !== existing.status) {
    await logActivity(userId, {
      eventType: `TASK_${input.status}`,
      description: `Task "${updated.title}" status changed to ${input.status}`,
      projectId: updated.projectId,
      taskId: updated.id,
    });
  }

  return updated;
}

/**
 * Deletes a task with ownership verification.
 */
export async function deleteTask(userId: string, taskId: string) {
  const existing = await prisma.task.findFirst({
    where: { id: taskId, userId },
    select: { id: true, title: true, projectId: true },
  });

  if (!existing) {
    throw new Error("Task not found or access denied.");
  }

  await prisma.task.delete({
    where: { id: taskId },
  });

  await logActivity(userId, {
    eventType: "TASK_DELETED",
    description: `Deleted task "${existing.title}"`,
    projectId: existing.projectId,
    taskId: null,
  });

  return { success: true, id: taskId };
}
