import { NextResponse } from "next/server";
import { modelRuntime } from "@/lib/ai/model-runtime";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  const info = await modelRuntime.getModelInfo();
  if (!info) {
    return NextResponse.json(
      {
        error: {
          code: "MODEL_NOT_LOADED",
          message: "No model is currently loaded in the local runtime.",
        },
      },
      { status: 503 },
    );
  }

  return NextResponse.json({
    name: info.name,
    architecture: info.architecture,
    parameters: info.parameters,
    contextLength: info.contextLength,
    device: info.device,
    precision: info.precision,
    vocabSize: info.vocabSize,
    loaded: info.loaded,
    uptimeSeconds: info.uptimeSeconds,
  });
}
