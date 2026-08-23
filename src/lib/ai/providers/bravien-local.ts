/**
 * Bravien local runtime client.
 *
 * Talks to the Bravien inference server (`bravien.inference.server`) over HTTP on
 * loopback. That server loads a Bravien checkpoint and runs Bravien's own weights;
 * there is no hosted model service anywhere in this path (§2, §76).
 *
 * The runtime's SSE frames already use the `AIStreamChunk` shapes, so streaming is
 * a parse-and-forward rather than a translation.
 */

import type {
  AIMessage,
  AIModel,
  AIStreamChunk,
  CompleteTextParams,
  StreamTextParams,
} from "@/types";

/** Where the local runtime listens. Loopback by default — never a remote host. */
export const RUNTIME_BASE_URL = (
  process.env.BRAVIEN_RUNTIME_URL?.trim() || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

/** Health checks must fail fast: the UI shows runtime status on every load. */
const HEALTH_TIMEOUT_MS = 2_500;
/** Generation can legitimately take a while on CPU. */
const GENERATION_TIMEOUT_MS = Number(
  process.env.BRAVIEN_RUNTIME_TIMEOUT_MS ?? 180_000,
);

export class RuntimeUnavailableError extends Error {
  readonly code = "RUNTIME_UNAVAILABLE" as const;

  constructor(message: string) {
    super(message);
    this.name = "RuntimeUnavailableError";
  }
}

export class RuntimeRequestError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(message: string, code: string, status: number) {
    super(message);
    this.name = "RuntimeRequestError";
    this.code = code;
    this.status = status;
  }
}

export interface RuntimeHealth {
  status: "ok" | "no_model" | "unreachable";
  modelLoaded: boolean;
  model: string | null;
  checkpoint: string | null;
  device: string | null;
  contextLength: number | null;
  error: string | null;
  backend: string;
  uptimeSeconds: number | null;
}

/** Model description as reported by the runtime. Every field is read, not assumed. */
export interface RuntimeModelInfo {
  name: string;
  version: string;
  architecture: string;
  parameters: number;
  layers: number;
  hidden_size: number;
  heads: number;
  kv_heads: number;
  context_length: number;
  vocab_size: number;
  norm: string;
  position_encoding: string;
  activation: string;
  precision: string;
  device: string;
  checkpoint: string | null;
  training: {
    stage: string;
    step: number;
    tokens_seen: number;
    run_name: string;
    created_at: string;
    metrics: Record<string, number>;
    dataset: Record<string, unknown>;
  };
  tokenizer: { vocab_size: number; checksum: string | null };
  /** Counters since this runtime process started. */
  served_generations?: number;
  served_completion_tokens?: number;
  uptime_seconds?: number;
}

interface RuntimeErrorBody {
  error?: { code?: string; message?: string };
  detail?: unknown;
}

async function runtimeFetch(
  path: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  // Chain the caller's signal so a client disconnect aborts the upstream request
  // instead of leaving the runtime generating into a void.
  const external = init.signal;
  if (external) {
    if (external.aborted) controller.abort();
    else external.addEventListener("abort", () => controller.abort(), { once: true });
  }

  try {
    return await fetch(`${RUNTIME_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      cache: "no-store",
    });
  } catch (cause) {
    const reason = cause instanceof Error ? cause.message : String(cause);
    throw new RuntimeUnavailableError(
      `Bravien runtime at ${RUNTIME_BASE_URL} is not reachable (${reason}). ` +
        `Start it with: python scripts/serve.py`,
    );
  } finally {
    clearTimeout(timer);
  }
}

async function throwForStatus(response: Response): Promise<never> {
  let code = "runtime_error";
  let message = `Bravien runtime returned ${response.status}`;
  try {
    const body = (await response.json()) as RuntimeErrorBody;
    if (body.error?.message) {
      code = body.error.code ?? code;
      message = body.error.message;
    } else if (body.detail) {
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch {
    // Body was not JSON; the status line is all we have.
  }
  throw new RuntimeRequestError(message, code, response.status);
}

/** Runtime status. Never throws — the UI needs a value to render either way. */
export async function getRuntimeHealth(): Promise<RuntimeHealth> {
  try {
    const response = await runtimeFetch("/health", { method: "GET" }, HEALTH_TIMEOUT_MS);
    if (!response.ok) {
      return {
        status: "unreachable",
        modelLoaded: false,
        model: null,
        checkpoint: null,
        device: null,
        contextLength: null,
        error: `runtime returned ${response.status}`,
        backend: "bravien-local",
        uptimeSeconds: null,
      };
    }
    const body = await response.json();
    return {
      status: body.status === "ok" ? "ok" : "no_model",
      modelLoaded: Boolean(body.model_loaded),
      model: body.model ?? null,
      checkpoint: body.checkpoint ?? null,
      device: body.device ?? null,
      contextLength: body.context_length ?? null,
      error: body.error ?? null,
      backend: body.backend ?? "bravien-local",
      uptimeSeconds: body.uptime_seconds ?? null,
    };
  } catch (error) {
    return {
      status: "unreachable",
      modelLoaded: false,
      model: null,
      checkpoint: null,
      device: null,
      contextLength: null,
      error: error instanceof Error ? error.message : String(error),
      backend: "bravien-local",
      uptimeSeconds: null,
    };
  }
}

/** Models the runtime is actually serving. Empty when no checkpoint is loaded. */
export async function listRuntimeModels(): Promise<RuntimeModelInfo[]> {
  const response = await runtimeFetch("/v1/models", { method: "GET" }, HEALTH_TIMEOUT_MS);
  if (!response.ok) await throwForStatus(response);
  const body = await response.json();
  return Array.isArray(body?.data) ? (body.data as RuntimeModelInfo[]) : [];
}

/**
 * Turn what the runtime reports into the UI's model shape.
 *
 * Capabilities list only `text`: this architecture has no vision tower and no
 * trained tool-calling, and claiming otherwise in the picker would be a lie the
 * user only discovers by trying it (§71, §72).
 */
export function toAIModel(info: RuntimeModelInfo): AIModel {
  const step = info.training?.step ?? 0;
  const stage = info.training?.stage ?? "untrained";
  const millions = info.parameters / 1_000_000;
  const size =
    millions >= 1000
      ? `${(millions / 1000).toFixed(1)}B`
      : `${millions.toFixed(1)}M`;

  return {
    id: info.name,
    name: info.name,
    description:
      `${size} parameters, ${info.layers} layers, ${info.context_length}-token context. ` +
      `${stage} checkpoint at step ${step.toLocaleString()}, running locally on ${info.device}.`,
    providerModelId: info.name,
    provider: "bravien-local",
    capabilities: ["text"],
    contextWindow: info.context_length,
    maxOutputTokens: Math.max(1, Math.min(4096, Math.floor(info.context_length / 2))),
    isDefault: true,
  };
}

/**
 * The runtime's own accounting for a prompt (§Phase 7).
 *
 * Every field is a real token count from the checkpoint's tokenizer. This is what
 * the UI displays; nothing here is estimated on the web side.
 */
export interface RuntimeContextPlan {
  input_tokens: number;
  max_context_tokens: number;
  reserved_output_tokens: number;
  available_tokens: number;
  overhead_tokens: number;
  truncated: boolean;
  truncated_turns: number;
  system_truncated: boolean;
  latest_user_truncated: boolean;
}

export interface RuntimeTokenCount {
  tokens: number;
  characters: number;
  max_context_tokens: number;
  fits_context: boolean;
  characters_per_token: number;
  token_ids?: number[];
  context?: RuntimeContextPlan;
}

/**
 * Exact token counts from the runtime's tokenizer.
 *
 * The only correct way for the web side to learn a token count: the tokenizer is
 * a property of the checkpoint, so any local approximation drifts the moment the
 * checkpoint changes. Costs a loopback round trip, which is the right price for a
 * number the user is shown.
 */
export async function countRuntimeTokens(params: {
  text?: string;
  messages?: AIMessage[];
  maxTokens?: number;
  signal?: AbortSignal;
}): Promise<RuntimeTokenCount> {
  const response = await runtimeFetch(
    "/v1/tokenize",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...(params.text !== undefined ? { text: params.text } : {}),
        ...(params.messages !== undefined
          ? {
              messages: params.messages.map((m) => ({
                role: m.role,
                content: m.content,
              })),
            }
          : {}),
        ...(typeof params.maxTokens === "number"
          ? { max_tokens: params.maxTokens }
          : {}),
      }),
      signal: params.signal,
    },
    HEALTH_TIMEOUT_MS,
  );

  if (!response.ok) await throwForStatus(response);
  return (await response.json()) as RuntimeTokenCount;
}

interface SamplingBody {
  max_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  seed?: number;
  stop?: string[];
}

function samplingFrom(params: StreamTextParams | CompleteTextParams): SamplingBody {
  const body: SamplingBody = {};
  if (typeof params.temperature === "number") body.temperature = params.temperature;
  if (typeof params.maxTokens === "number") body.max_tokens = params.maxTokens;
  return body;
}

/**
 * Parse an SSE byte stream into frames.
 *
 * Buffers across chunk boundaries: a `data:` line can be split mid-JSON by the
 * transport, and parsing per-chunk would drop those frames.
 */
async function* readSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<string, void, unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const line of frame.split("\n")) {
          if (line.startsWith("data: ")) yield line.slice(6);
        }
        boundary = buffer.indexOf("\n\n");
      }
    }
    // A final frame with no trailing blank line.
    const tail = buffer.trim();
    if (tail.startsWith("data: ")) yield tail.slice(6);
  } finally {
    reader.releaseLock();
  }
}

export class BravienLocalProvider {
  readonly id = "bravien-local";

  /**
   * Stream a chat completion from the local runtime.
   *
   * Yields the runtime's own frames. No delta is generated, reshaped or paced
   * here — what the model produced is what the UI receives (§52).
   */
  async *streamText(params: StreamTextParams): AsyncIterable<AIStreamChunk> {
    const response = await runtimeFetch(
      "/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: params.messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          stream: true,
          ...samplingFrom(params),
        }),
        signal: params.signal,
      },
      GENERATION_TIMEOUT_MS,
    );

    if (!response.ok) await throwForStatus(response);
    if (!response.body) {
      throw new RuntimeUnavailableError("runtime returned a streaming response with no body");
    }

    let sawComplete = false;
    for await (const data of readSSE(response.body)) {
      if (data === "[DONE]") break;

      let chunk: AIStreamChunk;
      try {
        chunk = JSON.parse(data) as AIStreamChunk;
      } catch {
        // A malformed frame is a runtime bug, not model output. Skip it rather
        // than surfacing raw JSON as if the model had written it.
        continue;
      }
      if (chunk.kind === "message_complete") sawComplete = true;
      yield chunk;
    }

    if (!sawComplete) {
      // The stream ended without a completion frame: the connection dropped
      // mid-generation. Say so instead of letting a truncated answer look final.
      yield {
        kind: "error",
        code: "stream_truncated",
        message: "The runtime closed the stream before the response finished.",
      };
    }
  }

  /** Non-streaming completion, for callers that just want the text. */
  async completeText(params: CompleteTextParams): Promise<string> {
    const response = await runtimeFetch(
      "/v1/chat/completions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: params.messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          stream: false,
          ...samplingFrom(params),
        }),
        signal: params.signal,
      },
      GENERATION_TIMEOUT_MS,
    );

    if (!response.ok) await throwForStatus(response);
    const body = await response.json();
    return typeof body?.text === "string" ? body.text : "";
  }
}
