import { NextResponse } from "next/server";
import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { rejectActionProposal } from "@/lib/ai/action-proposal";
import { badRequest, jsonError, unauthorized } from "@/lib/security/errors";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface Params {
  params: Promise<{ id: string }>;
}

export async function POST(_request: Request, { params }: Params) {
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

  try {
    const rejected = await rejectActionProposal(userId, id);
    return NextResponse.json({ proposal: rejected });
  } catch (error) {
    return badRequest(error instanceof Error ? error.message : "Rejection failed.");
  }
}
