/**
 * Provider resolution.
 *
 * There is exactly one provider: the local Bravien runtime. This file exists to
 * turn "the runtime is not running" or "that checkpoint is not loaded" into a
 * clear error, not to choose between backends.
 *
 * A hosted model service must never appear here. If Bravien cannot answer from
 * its own weights, the honest outcome is an error the user can act on (§2, §76).
 */

import { getModel, getRuntimeStatus } from "@/lib/ai/models";
import { BravienLocalProvider } from "@/lib/ai/providers/bravien-local";
import type { AIProvider } from "@/types";

export class ModelUnavailableError extends Error {
  readonly code = "MODEL_UNAVAILABLE" as const;

  constructor(message: string) {
    super(message);
    this.name = "ModelUnavailableError";
  }
}

/** One client instance is enough — it holds no per-request state. */
const provider = new BravienLocalProvider();

/**
 * Resolve the provider for a model id, verifying the runtime is actually serving
 * that model first. Checking here means a chat turn fails before streaming
 * starts, rather than half-way through an answer.
 */
export async function getProviderForModel(
  modelId: string,
): Promise<AIProvider> {
  const model = await getModel(modelId);
  if (model) return provider;

  const health = await getRuntimeStatus();

  if (health.status === "unreachable") {
    throw new ModelUnavailableError(
      `The Bravien runtime is not running. Start it with \`python scripts/serve.py\` ` +
        `and reload. (${health.error ?? "no response"})`,
    );
  }
  if (!health.modelLoaded) {
    throw new ModelUnavailableError(
      `The Bravien runtime is running but has no checkpoint loaded` +
        `${health.error ? `: ${health.error}` : "."} ` +
        `Train one with \`python scripts/pretrain.py\` or point BRAVIEN_CHECKPOINT at an existing checkpoint.`,
    );
  }
  throw new ModelUnavailableError(
    `The runtime is serving ${health.model ? `"${health.model}"` : "a different model"}, not "${modelId}".`,
  );
}

/** The provider, with no model check. For callers that already resolved a model. */
export function getLocalProvider(): AIProvider {
  return provider;
}

export { BravienLocalProvider } from "@/lib/ai/providers/bravien-local";
export {
  RuntimeRequestError,
  RuntimeUnavailableError,
} from "@/lib/ai/providers/bravien-local";
export type { AIProvider } from "@/types";
