/**
 * Shapes the browser receives from Bravien's own route handlers.
 *
 * The runtime types are imported from the server client rather than re-declared:
 * a duplicated interface would drift, and drift here means the UI claiming
 * something about the checkpoint that the checkpoint never said (§71). These are
 * type-only imports, so no server module reaches the browser bundle.
 */

import type {
  RuntimeHealth,
  RuntimeModelInfo,
} from "@/lib/ai/providers/bravien-local";
import type {
  AIModel,
  ConversationDTO,
  FeatureFlags,
  MemoryDTO,
  MessageDTO,
} from "@/types";

export type { RuntimeHealth, RuntimeModelInfo };

/** GET /api/runtime */
export interface RuntimeSnapshot {
  runtime: RuntimeHealth;
  models: AIModel[];
  /** Architecture and training detail, or null when no checkpoint is loaded. */
  activeModel: RuntimeModelInfo | null;
  features: FeatureFlags;
  persistence: "database" | "ephemeral";
}

/** GET /api/conversations */
export interface ConversationListResponse {
  items: ConversationDTO[];
  nextCursor: string | null;
}

/** GET /api/conversations/[id] */
export interface ConversationDetailResponse {
  conversation: ConversationDTO;
  messages: MessageDTO[];
}

/** GET /api/memories */
export interface MemoryListResponse {
  items: MemoryDTO[];
}

/** POST /api/files */
export interface UploadResponse {
  id: string | null;
  filename: string;
  mimeType: string;
  size: number;
  characters: number;
  text: string;
  warning?: string;
}

export interface ApiErrorBody {
  error?: { code?: string; message?: string; details?: unknown };
}

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(message: string, code: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

/**
 * Read an error out of a failed response.
 *
 * Every Bravien route answers failures as `{ error: { code, message } }`, and the
 * message is written to be shown to a person — so it is surfaced verbatim rather
 * than replaced with a generic "something went wrong".
 */
export async function apiErrorFrom(response: Response): Promise<ApiError> {
  let code = "REQUEST_FAILED";
  let message = `Request failed (${response.status}).`;
  try {
    const body = (await response.json()) as ApiErrorBody;
    if (body.error?.message) message = body.error.message;
    if (body.error?.code) code = body.error.code;
  } catch {
    // Not JSON — most likely an HTML error page. The status is all there is.
  }
  return new ApiError(message, code, response.status);
}

/** Fetch JSON from a Bravien route, throwing `ApiError` on failure. */
export async function apiFetch<T>(
  input: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(input, {
    ...init,
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });
  if (!response.ok) throw await apiErrorFrom(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}
