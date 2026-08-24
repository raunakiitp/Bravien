/**
 * Intelligence Efficiency Telemetry & Reporting for Bravien (§Phase 12, Phase 14).
 *
 * Tracks local model call avoidance, deterministic shortcut executions,
 * cache hit rates, average token budgets, inference latency, and failure recovery.
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
  recoveryAttempts: number;
  recoverySuccesses: number;
  recoveryFailures: number;
  totalRecoveryLatencyMs: number;
  recoveryByCategory: Record<string, number>;
  strategiesUsed: Record<string, number>;
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
  recovery: {
    recoveryAttempts: number;
    recoverySuccesses: number;
    recoveryFailures: number;
    strategySuccessRate: string;
    averageRecoveryLatencyMs: number;
    recoveryByCategory: Record<string, number>;
    strategiesUsed: Record<string, number>;
  };
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
    recoveryAttempts: 0,
    recoverySuccesses: 0,
    recoveryFailures: 0,
    totalRecoveryLatencyMs: 0,
    recoveryByCategory: {},
    strategiesUsed: {},
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
   * Record a recovery execution attempt.
   */
  public recordRecovery(params: {
    category: string;
    strategyId: string;
    success: boolean;
    latencyMs?: number;
  }): void {
    this.metrics.recoveryAttempts++;
    if (params.success) {
      this.metrics.recoverySuccesses++;
    } else {
      this.metrics.recoveryFailures++;
    }

    if (typeof params.latencyMs === "number") {
      this.metrics.totalRecoveryLatencyMs += params.latencyMs;
    }

    this.metrics.recoveryByCategory[params.category] =
      (this.metrics.recoveryByCategory[params.category] ?? 0) + 1;
    this.metrics.strategiesUsed[params.strategyId] =
      (this.metrics.strategiesUsed[params.strategyId] ?? 0) + 1;
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

    const recAttempts = Math.max(1, this.metrics.recoveryAttempts);
    const strategySuccessRate =
      this.metrics.recoveryAttempts > 0
        ? `${((this.metrics.recoverySuccesses / recAttempts) * 100).toFixed(1)}%`
        : "N/A";
    const averageRecoveryLatencyMs =
      this.metrics.recoveryAttempts > 0
        ? Math.round(this.metrics.totalRecoveryLatencyMs / recAttempts)
        : 0;

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
      recovery: {
        recoveryAttempts: this.metrics.recoveryAttempts,
        recoverySuccesses: this.metrics.recoverySuccesses,
        recoveryFailures: this.metrics.recoveryFailures,
        strategySuccessRate,
        averageRecoveryLatencyMs,
        recoveryByCategory: { ...this.metrics.recoveryByCategory },
        strategiesUsed: { ...this.metrics.strategiesUsed },
      },
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
      recoveryAttempts: 0,
      recoverySuccesses: 0,
      recoveryFailures: 0,
      totalRecoveryLatencyMs: 0,
      recoveryByCategory: {},
      strategiesUsed: {},
    };
  }
}

export const efficiencyTracker = EfficiencyTracker.getInstance();
