/**
 * Intelligence Efficiency Telemetry & Reporting for Bravien.
 *
 * Tracks local model call avoidance, deterministic shortcut executions,
 * cache hit rates, average token budgets, and inference latency.
 */

export interface EfficiencyMetrics {
  totalRequests: number;
  modelCalls: number;
  modelCallsAvoided: number;
  deterministicResponses: number;
  cacheHits: number;
  cacheMisses: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalLatencyMs: number;
}

export interface EfficiencyReport {
  totalRequests: number;
  modelCalls: number;
  modelCallsAvoided: number;
  deterministicRate: string;
  cacheHitRate: string;
  modelAvoidanceRate: string;
  averageInputTokens: number;
  averageOutputTokens: number;
  averageLatencyMs: number;
  totalGeneratedTokens: number;
}

export class EfficiencyTracker {
  private static instance: EfficiencyTracker | null = null;

  private metrics: EfficiencyMetrics = {
    totalRequests: 0,
    modelCalls: 0,
    modelCallsAvoided: 0,
    deterministicResponses: 0,
    cacheHits: 0,
    cacheMisses: 0,
    totalInputTokens: 0,
    totalOutputTokens: 0,
    totalLatencyMs: 0,
  };

  private constructor() {}

  public static getInstance(): EfficiencyTracker {
    if (!EfficiencyTracker.instance) {
      EfficiencyTracker.instance = new EfficiencyTracker();
    }
    return EfficiencyTracker.instance;
  }

  /**
   * Record a processed request turn.
   */
  public recordRequest(params: {
    avoidedModel: boolean;
    isDeterministic?: boolean;
    isCacheHit?: boolean;
    inputTokens?: number;
    outputTokens?: number;
    latencyMs?: number;
  }): void {
    this.metrics.totalRequests++;

    if (params.avoidedModel) {
      this.metrics.modelCallsAvoided++;
      if (params.isDeterministic) {
        this.metrics.deterministicResponses++;
      }
      if (params.isCacheHit) {
        this.metrics.cacheHits++;
      }
    } else {
      this.metrics.modelCalls++;
      this.metrics.cacheMisses++;
    }

    if (typeof params.inputTokens === "number") {
      this.metrics.totalInputTokens += params.inputTokens;
    }
    if (typeof params.outputTokens === "number") {
      this.metrics.totalOutputTokens += params.outputTokens;
    }
    if (typeof params.latencyMs === "number") {
      this.metrics.totalLatencyMs += params.latencyMs;
    }
  }

  /**
   * Generates a structured efficiency report.
   */
  public getEfficiencyReport(): EfficiencyReport {
    const total = Math.max(1, this.metrics.totalRequests);
    const modelCalls = Math.max(1, this.metrics.modelCalls);

    const deterministicRate = `${((this.metrics.deterministicResponses / total) * 100).toFixed(1)}%`;
    const cacheHitRate = `${((this.metrics.cacheHits / total) * 100).toFixed(1)}%`;
    const modelAvoidanceRate = `${((this.metrics.modelCallsAvoided / total) * 100).toFixed(1)}%`;

    const averageInputTokens = Math.round(this.metrics.totalInputTokens / total);
    const averageOutputTokens = Math.round(this.metrics.totalOutputTokens / modelCalls);
    const averageLatencyMs = Math.round(this.metrics.totalLatencyMs / total);

    return {
      totalRequests: this.metrics.totalRequests,
      modelCalls: this.metrics.modelCalls,
      modelCallsAvoided: this.metrics.modelCallsAvoided,
      deterministicRate,
      cacheHitRate,
      modelAvoidanceRate,
      averageInputTokens,
      averageOutputTokens,
      averageLatencyMs,
      totalGeneratedTokens: this.metrics.totalOutputTokens,
    };
  }

  public reset(): void {
    this.metrics = {
      totalRequests: 0,
      modelCalls: 0,
      modelCallsAvoided: 0,
      deterministicResponses: 0,
      cacheHits: 0,
      cacheMisses: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalLatencyMs: 0,
    };
  }
}

export const efficiencyTracker = EfficiencyTracker.getInstance();
