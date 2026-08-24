import { NextResponse } from "next/server";
import { modelRuntime } from "@/lib/ai/model-runtime";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  const health = await modelRuntime.healthCheck({ timeoutMs: 3000 });
  return NextResponse.json({
    status: health.status,
    modelLoaded: health.modelLoaded,
    model: health.model,
    device: health.device,
    contextLength: health.contextLength,
    backend: health.backend,
    uptimeSeconds: health.uptimeSeconds,
    runtimeState: modelRuntime.getState(),
  });
}
