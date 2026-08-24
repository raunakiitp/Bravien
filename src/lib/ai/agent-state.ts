/**
 * Persistent Agent State Layer for Bravien.
 *
 * Maintains explicit long-term agent state across conversations and tasks with
 * deterministic transitions, bounded arrays, and strict user ownership.
 */

import { z } from "zod";
import { prisma } from "@/lib/db/prisma";
import { logger } from "@/lib/observability/logger";
import { logActivity } from "@/lib/tasks/service";
import type { Prisma, AgentStateStatus } from "@prisma/client";

export const AgentStateStatusEnum = z.enum([
  "IDLE",
  "PLANNING",
  "EXECUTING",
  "WAITING_CONFIRMATION",
  "BLOCKED",
  "COMPLETED",
  "FAILED",
]);

export const MAX_CONSTRAINTS = 10;
export const MAX_DECISIONS = 15;
export const MAX_COMPLETED_ACTIONS = 20;
export const MAX_EVIDENCE_REFS = 10;

export interface EvidenceRef {
  title: string;
  url?: string;
  sourceType: "WEB" | "DOCUMENT" | "MEMORY" | "TOOL";
  snippet?: string;
  timestamp: string;
}

export type AgentStateDTO = Prisma.AgentStateGetPayload<{
  include: { actionProposals?: true; project?: true; task?: true };
}>;

export const createAgentStateSchema = z.object({
  goal: z.string().min(1, "Goal is required").max(1000, "Goal too long"),
  projectId: z.string().cuid().nullable().optional(),
  taskId: z.string().cuid().nullable().optional(),
  status: AgentStateStatusEnum.optional().default("IDLE"),
  constraints: z.array(z.string().max(200)).optional().default([]),
  decisions: z.array(z.string().max(200)).optional().default([]),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type CreateAgentStateInput = z.input<typeof createAgentStateSchema>;

export const updateAgentStateSchema = z.object({
  goal: z.string().min(1).max(1000).optional(),
  status: AgentStateStatusEnum.optional(),
  currentStep: z.number().int().min(1).max(20).optional(),
  constraints: z.array(z.string().max(200)).optional(),
  decisions: z.array(z.string().max(200)).optional(),
  completedActions: z.array(z.string().max(200)).optional(),
  evidenceRefs: z.array(z.any()).optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type UpdateAgentStateInput = z.input<typeof updateAgentStateSchema>;

/**
 * Validates deterministic state transitions.
 */
export function isValidStateTransition(
  current: AgentStateStatus,
  next: AgentStateStatus,
): boolean {
  if (current === next) return true;

  const validTransitions: Record<AgentStateStatus, AgentStateStatus[]> = {
    IDLE: ["PLANNING", "EXECUTING", "FAILED"],
    PLANNING: ["EXECUTING", "WAITING_CONFIRMATION", "BLOCKED", "FAILED"],
    EXECUTING: ["WAITING_CONFIRMATION", "BLOCKED", "COMPLETED", "FAILED", "PLANNING"],
    WAITING_CONFIRMATION: ["EXECUTING", "BLOCKED", "FAILED", "CANCELLED" as unknown as AgentStateStatus],
    BLOCKED: ["PLANNING", "EXECUTING", "FAILED", "COMPLETED"],
    COMPLETED: ["IDLE", "PLANNING", "EXECUTING"],
    FAILED: ["IDLE", "PLANNING", "EXECUTING"],
  };

  return validTransitions[current]?.includes(next) ?? false;
}

/**
 * Normalizes a rule/constraint/decision string.
 */
function normalizeFactString(str: string): string {
  return str
    .trim()
    .replace(/^[-*•\s]+/, "")
    .replace(/\s+/g, " ")
    .slice(0, 200);
}

/**
 * Gets or creates the active agent state for a user/project context.
 */
export async function getOrCreateActiveAgentState(
  userId: string,
  options: { projectId?: string | null; taskId?: string | null; goal?: string } = {},
) {
  // Check for existing active state
  const existing = await prisma.agentState.findFirst({
    where: {
      userId,
      ...(options.projectId ? { projectId: options.projectId } : {}),
      ...(options.taskId ? { taskId: options.taskId } : {}),
      status: { in: ["IDLE", "PLANNING", "EXECUTING", "WAITING_CONFIRMATION", "BLOCKED"] },
    },
    orderBy: { updatedAt: "desc" },
    include: {
      actionProposals: {
        where: { status: "PENDING" },
        orderBy: { createdAt: "desc" },
      },
    },
  });

  if (existing) return existing;

  // Create new active state
  const goal = options.goal?.trim() || "General conversational assistance";
  const newState = await prisma.agentState.create({
    data: {
      userId,
      projectId: options.projectId ?? null,
      taskId: options.taskId ?? null,
      goal,
      status: "IDLE",
      currentStep: 1,
      constraints: [],
      decisions: [],
      completedActions: [],
    },
    include: {
      actionProposals: true,
    },
  });

  await logActivity(userId, {
    eventType: "AGENT_STATE_CREATED",
    description: `Initialized agent state for: "${goal.slice(0, 60)}"`,
    projectId: newState.projectId,
    taskId: newState.taskId,
  });

  return newState;
}

/**
 * Fetches an agent state by ID with ownership verification.
 */
export async function getAgentStateById(userId: string, id: string) {
  const state = await prisma.agentState.findFirst({
    where: { id, userId },
    include: {
      project: { select: { id: true, name: true } },
      task: { select: { id: true, title: true, status: true } },
      actionProposals: { orderBy: { createdAt: "desc" } },
    },
  });

  if (!state) {
    throw new Error("Agent state not found or unauthorized access.");
  }

  return state;
}

/**
 * Retrieves the most relevant agent state for a context.
 */
export async function getRelevantAgentState(
  userId: string,
  options: { projectId?: string | null; taskId?: string | null } = {},
) {
  return prisma.agentState.findFirst({
    where: {
      userId,
      ...(options.taskId ? { taskId: options.taskId } : {}),
      ...(options.projectId ? { projectId: options.projectId } : {}),
    },
    orderBy: { updatedAt: "desc" },
    include: {
      actionProposals: {
        where: { status: "PENDING" },
      },
    },
  });
}

/**
 * Updates agent state with transition validation and array size bounds.
 */
export async function updateAgentState(
  userId: string,
  id: string,
  rawUpdates: UpdateAgentStateInput,
) {
  const existing = await prisma.agentState.findFirst({
    where: { id, userId },
  });

  if (!existing) {
    throw new Error("Agent state not found or access denied.");
  }

  const updates = updateAgentStateSchema.parse(rawUpdates);

  // Validate state transition if status changed
  if (updates.status && updates.status !== existing.status) {
    if (!isValidStateTransition(existing.status, updates.status as AgentStateStatus)) {
      throw new Error(
        `Invalid state transition from ${existing.status} to ${updates.status}.`,
      );
    }
  }

  const updated = await prisma.agentState.update({
    where: { id },
    data: {
      ...(updates.goal ? { goal: updates.goal } : {}),
      ...(updates.status ? { status: updates.status as AgentStateStatus } : {}),
      ...(updates.currentStep !== undefined ? { currentStep: updates.currentStep } : {}),
      ...(updates.constraints
        ? { constraints: updates.constraints.map(normalizeFactString).slice(0, MAX_CONSTRAINTS) }
        : {}),
      ...(updates.decisions
        ? { decisions: updates.decisions.map(normalizeFactString).slice(0, MAX_DECISIONS) }
        : {}),
      ...(updates.completedActions
        ? { completedActions: updates.completedActions.map(normalizeFactString).slice(0, MAX_COMPLETED_ACTIONS) }
        : {}),
      ...(updates.evidenceRefs
        ? { evidenceRefs: (Array.isArray(updates.evidenceRefs) ? updates.evidenceRefs.slice(0, MAX_EVIDENCE_REFS) : updates.evidenceRefs) as unknown as Prisma.InputJsonValue }
        : {}),
      ...(updates.metadata ? { metadata: updates.metadata as Prisma.InputJsonValue } : {}),
    },
    include: {
      actionProposals: true,
    },
  });

  if (updates.status && updates.status !== existing.status) {
    await logActivity(userId, {
      eventType: "AGENT_STATE_UPDATED",
      description: `Agent state transitioned to ${updates.status}`,
      projectId: updated.projectId,
      taskId: updated.taskId,
    });
  }

  return updated;
}

/**
 * Transitions agent state safely.
 */
export async function transitionState(
  userId: string,
  id: string,
  newStatus: AgentStateStatus,
) {
  return updateAgentState(userId, id, { status: newStatus });
}

/**
 * Adds a constraint to state with deduplication.
 */
export async function addConstraint(userId: string, id: string, constraint: string) {
  const norm = normalizeFactString(constraint);
  if (!norm) return null;

  const current = await getAgentStateById(userId, id);
  if (current.constraints.includes(norm)) return current;

  const newConstraints = [...current.constraints, norm].slice(-MAX_CONSTRAINTS);
  return updateAgentState(userId, id, { constraints: newConstraints });
}

/**
 * Records a decision in state with deduplication.
 */
export async function recordDecision(userId: string, id: string, decision: string) {
  const norm = normalizeFactString(decision);
  if (!norm) return null;

  const current = await getAgentStateById(userId, id);
  if (current.decisions.includes(norm)) return current;

  const newDecisions = [...current.decisions, norm].slice(-MAX_DECISIONS);
  return updateAgentState(userId, id, { decisions: newDecisions });
}

/**
 * Records a completed action.
 */
export async function recordCompletedAction(userId: string, id: string, action: string) {
  const norm = normalizeFactString(action);
  if (!norm) return null;

  const current = await getAgentStateById(userId, id);
  const newActions = [...current.completedActions, norm].slice(-MAX_COMPLETED_ACTIONS);
  return updateAgentState(userId, id, { completedActions: newActions });
}

/**
 * Records an evidence reference in state with deduplication.
 */
export async function recordEvidenceRef(
  userId: string,
  id: string,
  evidence: EvidenceRef,
) {
  const current = await getAgentStateById(userId, id);
  const existingRefs = (Array.isArray(current.evidenceRefs) ? current.evidenceRefs : []) as unknown as EvidenceRef[];

  const key = evidence.url || evidence.title;
  if (existingRefs.some((r) => (r.url || r.title) === key)) {
    return current;
  }

  const updatedRefs = [...existingRefs, evidence].slice(-MAX_EVIDENCE_REFS);
  return prisma.agentState.update({
    where: { id },
    data: { evidenceRefs: updatedRefs as unknown as Prisma.InputJsonValue },
  });
}

/**
 * Produces an ultra-compact summary string for Qwen 0.5B context budgeting.
 * Strict token and character bounded.
 */
export function summarizeAgentStateForContext(
  state?: {
    goal: string;
    status: string;
    currentStep?: number | null;
    constraints?: string[];
    decisions?: string[];
    completedActions?: string[];
    actionProposals?: Array<{ actionType: string; description: string }>;
  } | null,
): string {
  if (!state) return "";

  const lines: string[] = [
    `=== ACTIVE AGENT STATE ===`,
    `GOAL: ${state.goal.slice(0, 150)}`,
    `STATUS: ${state.status}${state.currentStep ? ` (Step ${state.currentStep})` : ""}`,
  ];

  if (state.constraints?.length) {
    lines.push(`ACTIVE CONSTRAINTS:\n${state.constraints.slice(-4).map((c) => `- ${c}`).join("\n")}`);
  }

  if (state.decisions?.length) {
    lines.push(`KEY DECISIONS:\n${state.decisions.slice(-4).map((d) => `- ${d}`).join("\n")}`);
  }

  if (state.completedActions?.length) {
    lines.push(`COMPLETED ACTIONS:\n${state.completedActions.slice(-3).map((a) => `- ${a}`).join("\n")}`);
  }

  if (state.actionProposals?.length) {
    lines.push(
      `PENDING CONFIRMATIONS:\n${state.actionProposals.slice(-2).map((p) => `[!] ${p.actionType}: ${p.description}`).join("\n")}`,
    );
  }

  return lines.join("\n\n");
}
