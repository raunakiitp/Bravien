import type { AIMessage } from "@/types";

/** Rough token estimate: ~4 chars per token for English-ish text. */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

export function estimateMessagesTokens(messages: AIMessage[]): number {
  return messages.reduce((sum, m) => {
    let n = estimateTokens(m.content) + 4;
    if (m.toolCalls) {
      for (const tc of m.toolCalls) {
        n += estimateTokens(tc.name) + estimateTokens(tc.arguments) + 8;
      }
    }
    return sum + n;
  }, 0);
}

/**
 * Trims older conversation turns to fit an approximate token budget.
 * Keeps system messages, a summary placeholder when truncating, and the most recent turns.
 */
export function trimConversationHistory(
  messages: AIMessage[],
  maxTokensApprox: number,
): AIMessage[] {
  if (messages.length === 0) return [];
  if (estimateMessagesTokens(messages) <= maxTokensApprox) {
    return [...messages];
  }

  const systemMessages = messages.filter((m) => m.role === "system");
  const nonSystem = messages.filter((m) => m.role !== "system");

  const kept: AIMessage[] = [];
  let used = estimateMessagesTokens(systemMessages);

  for (let i = nonSystem.length - 1; i >= 0; i--) {
    const msg = nonSystem[i]!;
    const cost = estimateMessagesTokens([msg]);
    if (used + cost > maxTokensApprox && kept.length > 0) {
      break;
    }
    kept.unshift(msg);
    used += cost;
  }

  const droppedCount = nonSystem.length - kept.length;
  const result: AIMessage[] = [...systemMessages];

  if (droppedCount > 0) {
    result.push({
      role: "system",
      content: `[Earlier conversation truncated: ${droppedCount} older message(s) omitted for context length. Continue based on the recent turns below.]`,
    });
  }

  result.push(...kept);
  return result;
}
