import { NextResponse } from "next/server";
import { modelRuntime } from "@/lib/ai/model-runtime";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST() {
  const result = await modelRuntime.warmup({ timeoutMs: 15_000 });
  return NextResponse.json(result);
}
