import { prisma } from "@/lib/db/prisma";
import { logger } from "@/lib/observability/logger";
import type { AIMessage } from "@/types";

/**
 * Creates a structured summary of older turns when conversation history grows long.
 */
export function summarizeMessagesLocally(messages: AIMessage[]): string {
  if (messages.length === 0) return "";

  const userRequirements: string[] = [];
  const assistantKeyPoints: string[] = [];

  for (const m of messages) {
    if (m.role === "user") {
      const line = m.content.split("\n")[0].trim().slice(0, 100);
      if (line) userRequirements.push(`- User discussed: ${line}`);
    } else if (m.role === "assistant") {
      const line = m.content.split("\n")[0].trim().slice(0, 100);
      if (line) assistantKeyPoints.push(`- Assistant noted: ${line}`);
    }
  }

  return [
    "Prior Conversation Context Summary:",
    ...userRequirements.slice(-5),
    ...assistantKeyPoints.slice(-5),
  ].join("\n");
}

/**
 * Updates the conversation summary in the database if conversation turn count is high.
 */
export async function updateConversationSummaryIfNeeded(
  conversationId: string,
  messages: AIMessage[],
  threshold = 10,
): Promise<string | null> {
  if (messages.length < threshold) return null;

  try {
    const summary = summarizeMessagesLocally(messages.slice(0, -4));
    await prisma.conversation.update({
      where: { id: conversationId },
      data: { summary },
    });
    return summary;
  } catch (err) {
    logger.warn("summarizer.update_failed", {
      error: err instanceof Error ? err.message : String(err),
    });
    return null;
  }
}
