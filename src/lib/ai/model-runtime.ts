/**
 * Bravien Model Runtime & Local Inference Engine Abstraction.
 *
 * Provides a clean, framework-agnostic interface to the local inference runtime:
 * - Lazy model loading & caching
 * - Concurrency control & generation slot management
 * - Device detection & model metadata inspection
 * - Warmup & health checks
 * - Token streaming with graceful cancellation
 * - Strict error sanitation (no secret leaks)
 */

import { logger } from "@/lib/observability/logger";
import {
  getRuntimeHealth,
  listRuntimeModels,
  type RuntimeHealth,
  type RuntimeModelInfo,
} from "./providers/bravien-local";
import { getLocalProvider } from "./providers";
import type {
  AIMessage,
  AIStreamChunk,
  CompleteTextParams,
  StreamTextParams,
} from "@/types";

export type RuntimeState = "IDLE" | "LOADING" | "READY" | "BUSY" | "FAILED";

export interface ModelRuntimeInfo {
  name: string;
  architecture: string;
  parameters: number;
  contextLength: number;
  device: string;
  precision: string;
  loaded: boolean;
  vocabSize: number;
  uptimeSeconds?: number;
}

export interface ModelGenerateParams {
  messages: AIMessage[];
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stop?: string[];
  signal?: AbortSignal;
}

export class ModelRuntime {
  private static instance: ModelRuntime | null = null;
  private state: RuntimeState = "IDLE";
  private activeModel: string | null = null;
  private loadPromise: Promise<boolean> | null = null;
  private isWarmedUp: boolean = false;
  private maxConcurrency: number = 1;
  private activeGenerations: number = 0;

  private constructor() {}

  public static getInstance(): ModelRuntime {
    if (!ModelRuntime.instance) {
      ModelRuntime.instance = new ModelRuntime();
    }
    return ModelRuntime.instance;
  }

  /**
   * Current operational state of the local model runtime.
   */
  public getState(): RuntimeState {
    return this.state;
  }

  /**
   * Whether a valid model is currently loaded and ready for inference.
   */
  public isLoaded(): boolean {
    return this.state === "READY" || this.state === "BUSY";
  }

  /**
   * Check runtime health status. Fast, non-blocking check with sanitized output.
   */
  public async healthCheck(options: { timeoutMs?: number } = {}): Promise<RuntimeHealth> {
    try {
      const health = await getRuntimeHealth();
      if (health.modelLoaded) {
        this.state = this.activeGenerations > 0 ? "BUSY" : "READY";
        this.activeModel = health.model;
      } else {
        this.state = "IDLE";
      }
      return health;
    } catch (err) {
      this.state = "FAILED";
      logger.warn("model_runtime.health_check_failed", {
        error: err instanceof Error ? err.message : String(err),
      });
      return {
        status: "unreachable",
        modelLoaded: false,
        model: null,
        checkpoint: null,
        device: null,
        contextLength: null,
        error: "Runtime service unreachable",
        backend: "bravien-local",
        uptimeSeconds: null,
      };
    }
  }

  /**
   * Retrieves sanitized metadata describing the active model and hardware.
   */
  public async getModelInfo(): Promise<ModelRuntimeInfo | null> {
    try {
      const models = await listRuntimeModels();
      if (!models || models.length === 0) return null;
      const info = models[0];

      return {
        name: info.name,
        architecture: info.architecture,
        parameters: info.parameters,
        contextLength: info.context_length,
        device: info.device,
        precision: info.precision,
        loaded: true,
        vocabSize: info.vocab_size,
        uptimeSeconds: info.uptime_seconds,
      };
    } catch (err) {
      logger.warn("model_runtime.get_info_failed", {
        error: err instanceof Error ? err.message : String(err),
      });
      return null;
    }
  }

  /**
   * Lazy load the model runtime, ensuring duplicate concurrent load requests are coalesced.
   */
  public loadModel(
    modelNameOrPath?: string,
    options: { timeoutMs?: number } = {},
  ): Promise<boolean> {
    if (this.loadPromise) {
      return this.loadPromise;
    }

    if (this.isLoaded() && (!modelNameOrPath || this.activeModel === modelNameOrPath)) {
      return Promise.resolve(true);
    }

    this.state = "LOADING";
    this.loadPromise = (async () => {
      try {
        const health = await this.healthCheck(options);
        if (health.modelLoaded) {
          this.state = "READY";
          this.activeModel = health.model;
          return true;
        }

        // If runtime service is up but no model loaded yet
        this.state = "IDLE";
        return false;
      } catch (err) {
        this.state = "FAILED";
        logger.error("model_runtime.load_failed", {
          error: err instanceof Error ? err.message : String(err),
        });
        return false;
      } finally {
        this.loadPromise = null;
      }
    })();

    return this.loadPromise;
  }

  /**
   * Warms up the model kernels with a tiny inference step.
   */
  public async warmup(options: { testPrompt?: string; timeoutMs?: number } = {}): Promise<{
    warmedUp: boolean;
    latencyMs: number;
    model: string | null;
    device: string | null;
  }> {
    const t0 = Date.now();
    try {
      const isReady = await this.loadModel(undefined, { timeoutMs: options.timeoutMs });
      if (!isReady) {
        return {
          warmedUp: false,
          latencyMs: Date.now() - t0,
          model: null,
          device: null,
        };
      }

      const info = await this.getModelInfo();
      const provider = getLocalProvider();
      if (provider.completeText) {
        await provider.completeText({
          model: info?.name ?? "bravien-local",
          messages: [{ role: "user", content: options.testPrompt ?? "Hello" }],
          maxTokens: 4,
          temperature: 0.0,
        });
      }

      this.isWarmedUp = true;
      return {
        warmedUp: true,
        latencyMs: Date.now() - t0,
        model: info?.name ?? null,
        device: info?.device ?? null,
      };
    } catch (err) {
      logger.warn("model_runtime.warmup_failed", {
        error: err instanceof Error ? err.message : String(err),
      });
      return {
        warmedUp: false,
        latencyMs: Date.now() - t0,
        model: this.activeModel,
        device: null,
      };
    }
  }

  /**
   * Synchronous-style text completion through the model runtime.
   */
  public async generate(params: ModelGenerateParams): Promise<string> {
    if (this.activeGenerations >= this.maxConcurrency) {
      throw new Error("Local model is busy processing maximum concurrent generations.");
    }

    this.activeGenerations++;
    this.state = "BUSY";

    try {
      const modelId = this.activeModel ?? "bravien-local";
      const completeParams: CompleteTextParams = {
        model: modelId,
        messages: params.messages,
        maxTokens: params.maxTokens,
        temperature: params.temperature,
        signal: params.signal,
      };

      const provider = getLocalProvider();
      if (!provider.completeText) {
        throw new Error("Local provider does not support completeText");
      }
      const result = await provider.completeText(completeParams);
      return result;
    } finally {
      this.activeGenerations = Math.max(0, this.activeGenerations - 1);
      this.state = this.activeGenerations > 0 ? "BUSY" : "READY";
    }
  }

  /**
   * Stream incremental tokens from the model runtime.
   */
  public async *streamGenerate(
    params: ModelGenerateParams,
  ): AsyncIterable<AIStreamChunk> {
    if (this.activeGenerations >= this.maxConcurrency) {
      yield {
        kind: "error",
        code: "MODEL_BUSY",
        message: "Local model runtime is currently busy with another generation. Please retry in a moment.",
      };
      return;
    }

    this.activeGenerations++;
    this.state = "BUSY";

    try {
      const modelId = this.activeModel ?? "bravien-local";
      const streamParams: StreamTextParams = {
        model: modelId,
        messages: params.messages,
        maxTokens: params.maxTokens,
        temperature: params.temperature,
        signal: params.signal,
      };

      const provider = getLocalProvider();
      for await (const chunk of provider.streamText(streamParams)) {
        if (params.signal?.aborted) {
          break;
        }
        yield chunk;
      }
    } finally {
      this.activeGenerations = Math.max(0, this.activeGenerations - 1);
      this.state = this.activeGenerations > 0 ? "BUSY" : "READY";
    }
  }

  /**
   * Reset runtime state and clear active model reference.
   */
  public unloadModel(): void {
    this.state = "IDLE";
    this.activeModel = null;
    this.isWarmedUp = false;
    this.activeGenerations = 0;
  }
}

export const modelRuntime = ModelRuntime.getInstance();
