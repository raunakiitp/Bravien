import type { FeatureFlags } from "@/types";

function envFlag(key: string, defaultValue = false): boolean {
  const raw = process.env[key]?.trim().toLowerCase();
  if (raw === undefined || raw === "") return defaultValue;
  return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
}

/**
 * Feature flags from environment.
 *
 * A flag defaults to true only when the feature is actually implemented and
 * working. `vision`, `voice`, `image_generation` and `code_execution` have no
 * implementation behind them: the Bravien architecture has no vision tower, and
 * code execution is deliberately absent until there is a real sandbox (§54).
 * They default off so the UI never advertises a capability that does not exist
 * (§72).
 */
export function getFeatureFlags(): FeatureFlags {
  return {
    web_search: envFlag("FEATURE_WEB_SEARCH", Boolean(process.env.SEARCH_API_KEY)),
    file_uploads: envFlag("FEATURE_FILE_UPLOADS", true),
    memory: envFlag("FEATURE_MEMORY", true),

    // Not implemented. Enabling these does not create the capability.
    vision: false,
    voice: false,
    image_generation: false,
    code_execution: false,
  };
}

export function isFeatureEnabled(flag: keyof FeatureFlags): boolean {
  return getFeatureFlags()[flag];
}
