/**
 * Controlled Memory Promotion System for Bravien.
 *
 * Promotes durable facts and preferences from AgentState to long-term memory
 * while filtering out temporary execution states and strictly rejecting secrets.
 */

import { prisma } from "@/lib/db/prisma";
import { createMemory } from "@/lib/memory/service";
import { logActivity } from "@/lib/tasks/service";
import { logger } from "@/lib/observability/logger";
import { getAgentStateById } from "./agent-state";

export interface MemoryPromotionOptions {
  includeConstraints?: boolean;
  includeDecisions?: boolean;
  projectId?: string | null;
}

/**
 * Checks whether a text contains potential secrets or passwords.
 */
export function containsSecretPatterns(text: string): boolean {
  return (
    /\b(?:password|passwd|api_key|apikey|secret_key|private_key|auth_token|bearer\s+[a-zA-Z0-9_\-.]+)\b/i.test(
      text,
    ) ||
    /ghp_[a-zA-Z0-9]{36}/.test(text) ||
    /sk-[a-zA-Z0-9]{32,}/.test(text)
  );
}

/**
 * Promotes durable constraints and decisions to long-term persistent memory.
 */
export async function promoteStateToPersistentMemory(
  userId: string,
  stateId: string,
  options: MemoryPromotionOptions = {},
) {
  const state = await getAgentStateById(userId, stateId);
  const includeConstraints = options.includeConstraints ?? true;
  const includeDecisions = options.includeDecisions ?? true;
  const promotedMemories: Array<{ id: string; content: string; type: string }> = [];

  // 1. Promote durable constraints as PREFERENCE or INSTRUCTION
  if (includeConstraints && state.constraints && state.constraints.length > 0) {
    for (const c of state.constraints) {
      if (containsSecretPatterns(c)) continue; // Reject secrets
      if (c.length < 5) continue;

      try {
        const mem = await createMemory({
          userId,
          content: c,
          type: "INSTRUCTION",
          projectId: options.projectId ?? state.projectId,
        });
        promotedMemories.push({ id: mem.id, content: mem.content, type: mem.type });
      } catch (err) {
        logger.warn("memory_promotion.constraint_failed", {
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
  }

  // 2. Promote decisions as FACT
  if (includeDecisions && state.decisions && state.decisions.length > 0) {
    for (const d of state.decisions) {
      if (containsSecretPatterns(d)) continue; // Reject secrets
      if (d.length < 5) continue;

      try {
        const mem = await createMemory({
          userId,
          content: d,
          type: "FACT",
          projectId: options.projectId ?? state.projectId,
        });
        promotedMemories.push({ id: mem.id, content: mem.content, type: mem.type });
      } catch (err) {
        logger.warn("memory_promotion.decision_failed", {
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
  }

  if (promotedMemories.length > 0) {
    await logActivity(userId, {
      eventType: "MEMORY_PROMOTED",
      description: `Promoted ${promotedMemories.length} durable facts from agent state to persistent memory`,
      projectId: state.projectId,
      taskId: state.taskId,
      metadata: { count: promotedMemories.length },
    });
  }

  return {
    stateId,
    promotedCount: promotedMemories.length,
    memories: promotedMemories,
  };
}
