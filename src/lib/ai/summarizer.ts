import { prisma } from "@/lib/db/prisma";
import { logger } from "@/lib/observability/logger";
import type { AIMessage } from "@/types";

/**
 * Creates a structured summary of older turns when conversation history grows long.
 */
export function summarizeMessagesLocally(messages: AIMessage[]): string {
  if (messages.length === 0) return "";

  const goalsAndRequests: string[] = [];
  const decisionsAndFacts: string[] = [];
  const constraintsAndRules: string[] = [];

  for (const m of messages) {
    const text = m.content.trim();
    if (m.role === "user") {
      const firstLine = text.split("\n")[0].slice(0, 120);
      if (/\b(?:must|don't|do not|never|always|rule|constraint)\b/i.test(text)) {
        constraintsAndRules.push(`- Constraint: ${firstLine}`);
      } else {
        goalsAndRequests.push(`- Goal/Request: ${firstLine}`);
      }
    } else if (m.role === "assistant") {
      const firstLine = text.split("\n")[0].slice(0, 120);
      if (/\b(?:decided|agreed|implemented|verified|fixed|chosen)\b/i.test(text)) {
        decisionsAndFacts.push(`- Decision/Result: ${firstLine}`);
      } else {
        decisionsAndFacts.push(`- Key Point: ${firstLine}`);
      }
    }
  }

  const sections: string[] = ["Prior Conversation Context Summary:"];
  if (constraintsAndRules.length) {
    sections.push("Active Constraints & Guidelines:\n" + constraintsAndRules.slice(-4).join("\n"));
  }
  if (goalsAndRequests.length) {
    sections.push("Prior Goals & Topics:\n" + goalsAndRequests.slice(-4).join("\n"));
  }
  if (decisionsAndFacts.length) {
    sections.push("Key Decisions & Findings:\n" + decisionsAndFacts.slice(-4).join("\n"));
  }

  return sections.join("\n\n");
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
