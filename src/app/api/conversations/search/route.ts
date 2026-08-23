import { NextResponse } from "next/server";

import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { logger } from "@/lib/observability/logger";
import { searchConversations } from "@/lib/search/conversations";
import { jsonError, unauthorized } from "@/lib/security/errors";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(request: Request) {
  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(
      503,
      "PERSISTENCE_UNAVAILABLE",
      "Conversation search requires a connected database.",
    );
  }

  const { searchParams } = new URL(request.url);
  const q = searchParams.get("q") ?? "";

  try {
    const results = await searchConversations(userId, q, { limit: 30 });
    return NextResponse.json({
      items: results.map((r) => ({
        ...r,
        updatedAt: r.updatedAt.toISOString(),
      })),
    });
  } catch (error) {
    logger.error("conversations.search_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "SEARCH_ERROR", "Search failed.");
  }
}
