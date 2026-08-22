import type { AIModel } from "@/types";

function envOr(key: string, fallback: string): string {
  const value = process.env[key]?.trim();
  return value && value.length > 0 ? value : fallback;
}

const MODEL_REGISTRY: AIModel[] = [
  {
    id: "bravien-fast",
    name: "Bravien Fast",
    description: "Quick responses for everyday tasks and chat.",
    providerModelId: envOr("BRAVIEN_MODEL_FAST", "gpt-4o-mini"),
    provider: "openai-compatible",
    capabilities: ["text", "tools"],
    contextWindow: 128_000,
    maxOutputTokens: 4_096,
    isDefault: true,
  },
  {
    id: "bravien-balanced",
    name: "Bravien Balanced",
    description: "Strong all-around model for writing, analysis, and coding.",
    providerModelId: envOr("BRAVIEN_MODEL_BALANCED", "gpt-4o"),
    provider: "openai-compatible",
    capabilities: ["text", "tools", "vision"],
    contextWindow: 128_000,
    maxOutputTokens: 8_192,
  },
  {
    id: "bravien-reasoning",
    name: "Bravien Reasoning",
    description: "Deeper reasoning for complex problems and multi-step work.",
    providerModelId: envOr("BRAVIEN_MODEL_REASONING", "o3-mini"),
    provider: "openai-compatible",
    capabilities: ["text", "tools", "reasoning"],
    contextWindow: 200_000,
    maxOutputTokens: 16_384,
  },
  {
    id: "bravien-vision",
    name: "Bravien Vision",
    description: "Image understanding with multimodal context.",
    providerModelId: envOr("BRAVIEN_MODEL_VISION", "gpt-4o"),
    provider: "openai-compatible",
    capabilities: ["text", "vision", "tools"],
    contextWindow: 128_000,
    maxOutputTokens: 8_192,
  },
];

export function getModels(): AIModel[] {
  return MODEL_REGISTRY.map((m) => ({ ...m }));
}

export function getModel(id: string): AIModel | undefined {
  return MODEL_REGISTRY.find((m) => m.id === id);
}

export function getDefaultModelId(): string {
  const fromEnv = process.env.BRAVIEN_DEFAULT_MODEL?.trim();
  if (fromEnv && getModel(fromEnv)) return fromEnv;
  const marked = MODEL_REGISTRY.find((m) => m.isDefault);
  return marked?.id ?? "bravien-fast";
}
