import { NextResponse } from "next/server";
import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { resumeTask } from "@/lib/tasks/executor";
import { badRequest, jsonError, unauthorized } from "@/lib/security/errors";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface Params {
  params: Promise<{ id: string }>;
}

export async function POST(request: Request, { params }: Params) {
  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(503, "PERSISTENCE_UNAVAILABLE", "Database is unreachable.");
  }

  const { id } = await params;

  let timeoutMs: number | undefined;
  try {
    const body = await request.json();
    if (body && typeof body.timeoutMs === "number") {
      timeoutMs = body.timeoutMs;
    }
  } catch {
    // Optional
  }

  try {
    const result = await resumeTask(userId, id, { timeoutMs });
    return NextResponse.json({ result });
  } catch (error) {
    return badRequest(error instanceof Error ? error.message : "Failed to resume task.");
  }
}
