/**
 * Action Proposal & Human Confirmation Gate for Bravien.
 *
 * Implements deterministic action risk classification and confirmation gating.
 * Destructive/irreversible actions MUST be approved by the user before execution.
 */

import { z } from "zod";
import { prisma } from "@/lib/db/prisma";
import { logActivity } from "@/lib/tasks/service";
import type { Prisma, ActionRiskLevel, ActionProposalStatus } from "@prisma/client";

export const ActionRiskLevelEnum = z.enum(["LOW", "MEDIUM", "HIGH"]);
export const ActionProposalStatusEnum = z.enum(["PENDING", "APPROVED", "REJECTED", "EXECUTED"]);

export const createActionProposalSchema = z.object({
  actionType: z.string().min(1).max(100),
  description: z.string().min(1).max(1000),
  riskLevel: ActionRiskLevelEnum.optional(),
  agentStateId: z.string().cuid().nullable().optional(),
  taskId: z.string().cuid().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type CreateActionProposalInput = z.input<typeof createActionProposalSchema>;

/**
 * Deterministically classifies action risk.
 */
export function classifyActionRisk(actionType: string): {
  riskLevel: ActionRiskLevel;
  requiresConfirmation: boolean;
} {
  const norm = actionType.toLowerCase().trim();

  // High Risk / Destructive / Irreversible -> requires confirmation
  if (
    /^(?:delete|drop|truncate|destroy|remove|purge|wipe|reset|modify_credential|send_external|execute_shell)/i.test(
      norm,
    ) ||
    norm.includes("delete") ||
    norm.includes("drop") ||
    norm.includes("wipe") ||
    norm.includes("truncate")
  ) {
    return { riskLevel: "HIGH", requiresConfirmation: true };
  }

  // Medium Risk -> state modification
  if (
    /^(?:fetch_web_page|create_task|update_memory|modify_setting|execute_query)/i.test(
      norm,
    )
  ) {
    return { riskLevel: "MEDIUM", requiresConfirmation: false };
  }

  // Low Risk -> pure read-only / arithmetic
  return { riskLevel: "LOW", requiresConfirmation: false };
}

/**
 * Creates an action proposal.
 */
export async function createActionProposal(
  userId: string,
  rawInput: CreateActionProposalInput,
) {
  const input = createActionProposalSchema.parse(rawInput);
  const { riskLevel, requiresConfirmation } = classifyActionRisk(input.actionType);

  const finalRiskLevel = (input.riskLevel as ActionRiskLevel) || riskLevel;
  const finalRequiresConfirmation = finalRiskLevel === "HIGH" || requiresConfirmation;

  const proposal = await prisma.actionProposal.create({
    data: {
      userId,
      agentStateId: input.agentStateId ?? null,
      taskId: input.taskId ?? null,
      actionType: input.actionType,
      description: input.description,
      riskLevel: finalRiskLevel,
      status: "PENDING",
      requiresConfirmation: finalRequiresConfirmation,
      metadata: (input.metadata as Prisma.InputJsonValue) ?? undefined,
    },
  });

  await logActivity(userId, {
    eventType: finalRequiresConfirmation ? "CONFIRMATION_REQUESTED" : "ACTION_PROPOSED",
    description: `Action proposed: "${input.actionType}" (${finalRiskLevel} risk)`,
    taskId: input.taskId,
  });

  return proposal;
}

/**
 * Approves a pending action proposal with user ownership check.
 */
export async function approveActionProposal(userId: string, proposalId: string) {
  const proposal = await prisma.actionProposal.findFirst({
    where: { id: proposalId, userId, status: "PENDING" },
  });

  if (!proposal) {
    throw new Error("Pending action proposal not found or unauthorized.");
  }

  const updated = await prisma.actionProposal.update({
    where: { id: proposalId },
    data: { status: "APPROVED" },
  });

  await logActivity(userId, {
    eventType: "CONFIRMATION_APPROVED",
    description: `Approved action proposal "${proposal.actionType}"`,
    taskId: proposal.taskId,
  });

  return updated;
}

/**
 * Rejects a pending action proposal with user ownership check.
 */
export async function rejectActionProposal(userId: string, proposalId: string) {
  const proposal = await prisma.actionProposal.findFirst({
    where: { id: proposalId, userId, status: "PENDING" },
  });

  if (!proposal) {
    throw new Error("Pending action proposal not found or unauthorized.");
  }

  const updated = await prisma.actionProposal.update({
    where: { id: proposalId },
    data: { status: "REJECTED" },
  });

  await logActivity(userId, {
    eventType: "CONFIRMATION_REJECTED",
    description: `Rejected action proposal "${proposal.actionType}"`,
    taskId: proposal.taskId,
  });

  return updated;
}

/**
 * Lists pending action proposals for a user.
 */
export async function listPendingActionProposals(
  userId: string,
  options: { agentStateId?: string | null; taskId?: string | null } = {},
) {
  return prisma.actionProposal.findMany({
    where: {
      userId,
      status: "PENDING",
      ...(options.agentStateId ? { agentStateId: options.agentStateId } : {}),
      ...(options.taskId ? { taskId: options.taskId } : {}),
    },
    orderBy: { createdAt: "desc" },
  });
}
