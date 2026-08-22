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

  OPENAI_API_KEY: z.string().optional(),
  OPENAI_BASE_URL: z.string().optional(),
  ANTHROPIC_API_KEY: z.string().optional(),

  BRAVIEN_MODEL_FAST: z.string().optional(),
  BRAVIEN_MODEL_BALANCED: z.string().optional(),
  BRAVIEN_MODEL_REASONING: z.string().optional(),
  BRAVIEN_MODEL_VISION: z.string().optional(),
  BRAVIEN_DEFAULT_MODEL: z.string().optional(),

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
  openaiBaseUrl: string;
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
    openaiBaseUrl: data.OPENAI_BASE_URL || "https://api.openai.com/v1",
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
