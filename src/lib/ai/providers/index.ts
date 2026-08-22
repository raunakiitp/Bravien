import { getModel } from "@/lib/ai/models";
import { OpenAICompatibleProvider } from "@/lib/ai/providers/openai-compatible";
import type { AIProvider } from "@/types";

export class ModelUnavailableError extends Error {
  readonly code = "MODEL_UNAVAILABLE" as const;

  constructor(message: string) {
    super(message);
    this.name = "ModelUnavailableError";
  }
}

/**
 * Resolve a provider for a Bravien model id.
 * Currently OpenAI-compatible (OPENAI_API_KEY + OPENAI_BASE_URL).
 * Anthropic can be wired later when ANTHROPIC_API_KEY is set and a provider is implemented.
 */
export function getProviderForModel(modelId: string): AIProvider {
  const model = getModel(modelId);
  if (!model) {
    throw new ModelUnavailableError(`Unknown model: ${modelId}`);
  }

  if (model.provider === "anthropic") {
    const anthropicKey = process.env.ANTHROPIC_API_KEY?.trim();
    if (!anthropicKey) {
      throw new ModelUnavailableError(
        "MODEL_UNAVAILABLE: Anthropic provider is not configured (missing ANTHROPIC_API_KEY)",
      );
    }
    // Native Anthropic Messages API provider not implemented yet; fall through guidance.
    throw new ModelUnavailableError(
      "MODEL_UNAVAILABLE: Anthropic provider is configured for future use but not yet implemented. Use an openai-compatible model.",
    );
  }

  const apiKey = process.env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    throw new ModelUnavailableError(
      "MODEL_UNAVAILABLE: Missing OPENAI_API_KEY. Configure an OpenAI-compatible API key to use Bravien models.",
    );
  }

  const baseUrl =
    process.env.OPENAI_BASE_URL?.trim() || "https://api.openai.com/v1";

  return new OpenAICompatibleProvider({
    apiKey,
    baseUrl,
    id: "openai-compatible",
  });
}

export { OpenAICompatibleProvider } from "@/lib/ai/providers/openai-compatible";
export type { AIProvider } from "@/types";
