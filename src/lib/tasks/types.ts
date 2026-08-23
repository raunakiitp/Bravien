import { z } from "zod";
import type { TaskStatus, TaskPriority, TaskType } from "@prisma/client";

export const TaskStatusEnum = z.enum([
  "PENDING",
  "RUNNING",
  "COMPLETED",
  "FAILED",
  "CANCELLED",
]);

export const TaskPriorityEnum = z.enum(["LOW", "NORMAL", "HIGH", "URGENT"]);

export const TaskTypeEnum = z.enum([
  "RESEARCH",
  "ANALYSIS",
  "DOCUMENT",
  "CODING",
  "PROJECT",
  "GENERAL",
]);

export const createTaskSchema = z.object({
  title: z.string().min(1, "Title is required").max(200, "Title is too long"),
  description: z.string().max(4000, "Description is too long").optional(),
  type: TaskTypeEnum.optional().default("GENERAL"),
  priority: TaskPriorityEnum.optional().default("NORMAL"),
  projectId: z.string().cuid().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type CreateTaskInput = z.input<typeof createTaskSchema>;

export const updateTaskSchema = z.object({
  title: z.string().min(1).max(200).optional(),
  description: z.string().max(4000).nullable().optional(),
  type: TaskTypeEnum.optional(),
  status: TaskStatusEnum.optional(),
  priority: TaskPriorityEnum.optional(),
  resultSummary: z.string().max(10000).nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type UpdateTaskInput = z.input<typeof updateTaskSchema>;

export const taskFilterSchema = z.object({
  projectId: z.string().optional(),
  status: TaskStatusEnum.optional(),
  type: TaskTypeEnum.optional(),
  search: z.string().optional(),
  limit: z.coerce.number().int().min(1).max(100).optional().default(30),
  offset: z.coerce.number().int().min(0).optional().default(0),
});

export type TaskFilterInput = z.input<typeof taskFilterSchema>;

export interface TaskExecutionResult {
  taskId: string;
  executionId: string;
  status: TaskStatus;
  startedAt: string;
  completedAt?: string;
  durationMs: number;
  stepsCount: number;
  successfulSteps: number;
  resultSummary: string;
  evidenceCount?: number;
  citations?: Array<{ title: string; url: string; snippet: string }>;
  error?: string;
}

export interface ActivityEvent {
  id: string;
  userId: string;
  projectId?: string | null;
  taskId?: string | null;
  eventType: string;
  description: string;
  metadata?: Record<string, unknown>;
  createdAt: string;
}
