/**
 * Conversation sizing on the web side (§Phase 7).
 *
 * The authoritative context budget lives in the runtime, in
 * `bravien/inference/context.py`, where the real tokenizer is. That is not a
 * layering preference — it is the only place a token count can be correct. This
 * module used to divide characters by four and treat the result as the budget;
 * Bravien's byte-level BPE runs about 1.55 characters per token on English prose,
 * so that undercounted a 826-token prompt as 326 and the runtime then cut the
 * rest, system prompt included.
 *
 * What remains here is deliberately not a budget:
 *
 * * `approxTokenFloor` is an explicit *under*count, valid only for "this is at
 *   least N tokens" reasoning. It must never gate what gets sent.
 * * `boundHistoryForPayload` is a payload guard, not a trim. A long-running
 *   conversation should not ship megabytes of history for the runtime to
 *   tokenize and discard, but the guard keeps several times more than can
 *   possibly fit, so it never removes a turn the runtime would have kept.
 * * `countRuntimeTokens`, in the runtime client, asks the runtime for a real
 *   count. It is the only source of a token number worth displaying.
 *
 * The division of labour: this module bounds the payload, the runtime decides
 * what fits and reports it, and the UI shows the runtime's numbers.
 */

import type { AIMessage } from "@/types";

/**
 * A deliberate *lower* bound on the token count of some text.
 *
 * Four characters per token is roughly 2.6x too generous for Bravien's
 * tokenizer, which is exactly why this is named as a floor and used only where
 * undershooting is safe. For any number a user sees, or any decision about what
 * fits, call `countTokensExact` instead.
 */
export function approxTokenFloor(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

/** The same floor across a message list, including a little per-message framing. */
export function approxMessagesTokenFloor(messages: AIMessage[]): number {
  return messages.reduce((sum, m) => {
    let n = approxTokenFloor(m.content) + 4;
    if (m.toolCalls) {
      for (const tc of m.toolCalls) {
        n += approxTokenFloor(tc.name) + approxTokenFloor(tc.arguments) + 8;
      }
    }
    return sum + n;
  }, 0);
}

/**
 * How much more than the context window may be sent, in floor-estimated tokens.
 *
 * The floor undercounts by ~2.6x, so a 4x window of floor-tokens is comfortably
 * more than the runtime can accept however the text tokenizes. The guard exists
 * to stop unbounded payload growth, not to make anything fit.
 */
const PAYLOAD_SLACK = 4;

/**
 * Bound the history sent to the runtime, without deciding what fits.
 *
 * System messages and the newest turn always survive. Older turns are kept
 * newest-first while the floor estimate stays under the slack allowance. The
 * runtime then applies the real budget and reports what it dropped — so
 * truncation is surfaced to the user rather than announced to the model in an
 * injected note it is too small to read.
 */
export function boundHistoryForPayload(
  messages: AIMessage[],
  contextWindow: number,
): AIMessage[] {
  if (messages.length === 0) return [];

  const allowance = Math.max(1, contextWindow) * PAYLOAD_SLACK;
  if (approxMessagesTokenFloor(messages) <= allowance) {
    return [...messages];
  }

  const systemMessages = messages.filter((m) => m.role === "system");
  const nonSystem = messages.filter((m) => m.role !== "system");

  const kept: AIMessage[] = [];
  let used = approxMessagesTokenFloor(systemMessages);

  for (let i = nonSystem.length - 1; i >= 0; i--) {
    const msg = nonSystem[i]!;
    const cost = approxMessagesTokenFloor([msg]);
    // `kept.length > 0` keeps the newest turn whatever it costs: the runtime
    // shortens an oversized question intelligently, and dropping it here would
    // send a request with no question in it.
    if (used + cost > allowance && kept.length > 0) break;
    kept.unshift(msg);
    used += cost;
  }

  return [...systemMessages, ...kept];
}
