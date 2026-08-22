import type { FeatureFlags } from "@/types";

function envFlag(key: string, defaultValue = false): boolean {
  const raw = process.env[key]?.trim().toLowerCase();
  if (raw === undefined || raw === "") return defaultValue;
  return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
}

/**
 * Feature flags from environment.
 * Soft defaults: most optional features off until explicitly enabled.
 */
export function getFeatureFlags(): FeatureFlags {
  return {
    web_search: envFlag("FEATURE_WEB_SEARCH", Boolean(process.env.SEARCH_API_KEY)),
    file_uploads: envFlag("FEATURE_FILE_UPLOADS", true),
    vision: envFlag("FEATURE_VISION", true),
    voice: envFlag("FEATURE_VOICE", false),
    memory: envFlag("FEATURE_MEMORY", true),
    image_generation: envFlag("FEATURE_IMAGE_GENERATION", false),
    code_execution: envFlag("FEATURE_CODE_EXECUTION", false),
  };
}

export function isFeatureEnabled(flag: keyof FeatureFlags): boolean {
  return getFeatureFlags()[flag];
}
