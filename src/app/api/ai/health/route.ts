import { NextResponse } from "next/server";
import { modelRuntime } from "@/lib/ai/model-runtime";
import { efficiencyTracker } from "@/lib/ai/efficiency";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  const health = await modelRuntime.healthCheck({ timeoutMs: 3000 });
  const metrics = modelRuntime.getMetrics();
  const efficiency = efficiencyTracker.getEfficiencyReport();

  return NextResponse.json({
    status: health.status,
    modelLoaded: health.modelLoaded,
    model: health.model,
    device: health.device,
    contextLength: health.contextLength,
    backend: health.backend,
    uptimeSeconds: health.uptimeSeconds,
    runtimeState: modelRuntime.getState(),
    activeRequests: metrics.activeGenerations,
    metrics: {
      requestCount: metrics.requestCount,
      totalTokensGenerated: metrics.totalTokensGenerated,
      errorCount: metrics.errorCount,
      lastLatencyMs: metrics.lastLatencyMs,
      isWarmedUp: metrics.isWarmedUp,
    },
    efficiency: {
      totalRequests: efficiency.totalRequests,
      modelCalls: efficiency.modelCalls,
      modelCallsAvoided: efficiency.modelCallsAvoided,
      deterministicRate: efficiency.deterministicRate,
      cacheHitRate: efficiency.cacheHitRate,
      modelAvoidanceRate: efficiency.modelAvoidanceRate,
      averageInputTokens: efficiency.averageInputTokens,
      averageOutputTokens: efficiency.averageOutputTokens,
      averageLatencyMs: efficiency.averageLatencyMs,
    },
  });
}
