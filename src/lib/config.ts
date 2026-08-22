import { z } from "zod";

/**
 * Soft env config — optional keys are fine so `next build` succeeds without secrets.
 * Call `getConfig()` at runtime when values are needed.
 */
const envSchema = z.object({
  NODE_ENV: z.enum(["development", "test", "production"]).optional(),
  DATABASE_URL: z.string().optional(),
  AUTH_SECRET: z.string().optional(),
  AUTH_URL: z.string().optional(),

  /**
   * Bravien's inference backend. There are no model-service API keys here by
   * design: responses come from a local Bravien checkpoint served by
   * `bravien.inference.server`, or they do not come at all (§2, §76).
   */
  BRAVIEN_RUNTIME_URL: z.string().optional(),
  BRAVIEN_RUNTIME_TIMEOUT_MS: z.string().optional(),
  BRAVIEN_DEFAULT_MODEL: z.string().optional(),
  BRAVIEN_MODEL_TITLES: z.string().optional(),

  SEARCH_API_KEY: z.string().optional(),

  GOOGLE_CLIENT_ID: z.string().optional(),
  GOOGLE_CLIENT_SECRET: z.string().optional(),
  GITHUB_CLIENT_ID: z.string().optional(),
  GITHUB_CLIENT_SECRET: z.string().optional(),

  FEATURE_WEB_SEARCH: z.string().optional(),
  FEATURE_FILE_UPLOADS: z.string().optional(),
  FEATURE_VISION: z.string().optional(),
  FEATURE_VOICE: z.string().optional(),
  FEATURE_MEMORY: z.string().optional(),
  FEATURE_IMAGE_GENERATION: z.string().optional(),
  FEATURE_CODE_EXECUTION: z.string().optional(),

  UPLOAD_DIR: z.string().optional(),
  MAX_UPLOAD_BYTES: z.string().optional(),
});

export type AppConfig = z.infer<typeof envSchema> & {
  runtimeUrl: string;
  uploadDir: string;
  maxUploadBytes: number;
};

let cached: AppConfig | null = null;

export function getConfig(): AppConfig {
  if (cached) return cached;

  const parsed = envSchema.safeParse(process.env);
  const data = parsed.success ? parsed.data : {};

  const maxUploadBytes = Number(data.MAX_UPLOAD_BYTES ?? 20 * 1024 * 1024);

  cached = {
    ...data,
    runtimeUrl: (
      data.BRAVIEN_RUNTIME_URL || "http://127.0.0.1:8000"
    ).replace(/\/+$/, ""),
    uploadDir: data.UPLOAD_DIR || "uploads",
    maxUploadBytes: Number.isFinite(maxUploadBytes)
      ? maxUploadBytes
      : 20 * 1024 * 1024,
  };

  return cached;
}

/** Reset cached config (tests / hot reload). */
export function resetConfigCache(): void {
  cached = null;
}
