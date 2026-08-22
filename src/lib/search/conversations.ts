import { prisma } from "@/lib/db/prisma";

export interface ConversationSearchHit {
  conversationId: string;
  title: string;
  updatedAt: Date;
  snippet: string | null;
  matchSource: "title" | "message";
}

/**
 * Search conversations and message content for a user.
 * Uses PostgreSQL full-text search when available; falls back to ILIKE.
 */
export async function searchConversations(
  userId: string,
  query: string,
  options?: { limit?: number },
): Promise<ConversationSearchHit[]> {
  const q = query.trim();
  if (!q) return [];
  const limit = options?.limit ?? 25;

  // Prefer Postgres FTS via plainto_tsquery when the DB supports it.
  try {
    const fts = await prisma.$queryRaw<
      Array<{
        conversation_id: string;
        title: string;
        updated_at: Date;
        snippet: string | null;
        match_source: string;
      }>
    >`
      (
        SELECT
          c.id AS conversation_id,
          c.title AS title,
          c."updatedAt" AS updated_at,
          NULL::text AS snippet,
          'title' AS match_source
        FROM "Conversation" c
        WHERE c."userId" = ${userId}
          AND to_tsvector('english', coalesce(c.title, ''))
              @@ plainto_tsquery('english', ${q})
      )
      UNION ALL
      (
        SELECT
          c.id AS conversation_id,
          c.title AS title,
          c."updatedAt" AS updated_at,
          left(m.content, 200) AS snippet,
          'message' AS match_source
        FROM "Message" m
        INNER JOIN "Conversation" c ON c.id = m."conversationId"
        WHERE c."userId" = ${userId}
          AND to_tsvector('english', coalesce(m.content, ''))
              @@ plainto_tsquery('english', ${q})
        LIMIT ${limit}
      )
      ORDER BY updated_at DESC
      LIMIT ${limit}
    `;

    return fts.map((row) => ({
      conversationId: row.conversation_id,
      title: row.title,
      updatedAt: row.updated_at,
      snippet: row.snippet,
      matchSource: row.match_source === "title" ? "title" : "message",
    }));
  } catch {
    // ILIKE fallback (works without FTS indexes / on non-Postgres drivers in tests)
    return searchConversationsIlike(userId, q, limit);
  }
}

async function searchConversationsIlike(
  userId: string,
  query: string,
  limit: number,
): Promise<ConversationSearchHit[]> {
  const titleHits = await prisma.conversation.findMany({
    where: {
      userId,
      title: { contains: query, mode: "insensitive" },
    },
    orderBy: { updatedAt: "desc" },
    take: limit,
    select: { id: true, title: true, updatedAt: true },
  });

  const messageHits = await prisma.message.findMany({
    where: {
      conversation: { userId },
      content: { contains: query, mode: "insensitive" },
    },
    orderBy: { createdAt: "desc" },
    take: limit,
    select: {
      content: true,
      conversation: { select: { id: true, title: true, updatedAt: true } },
    },
  });

  const results: ConversationSearchHit[] = [
    ...titleHits.map((c) => ({
      conversationId: c.id,
      title: c.title,
      updatedAt: c.updatedAt,
      snippet: null,
      matchSource: "title" as const,
    })),
    ...messageHits.map((m) => ({
      conversationId: m.conversation.id,
      title: m.conversation.title,
      updatedAt: m.conversation.updatedAt,
      snippet: m.content.slice(0, 200),
      matchSource: "message" as const,
    })),
  ];

  // Dedupe by conversation, prefer title matches
  const seen = new Set<string>();
  const deduped: ConversationSearchHit[] = [];
  for (const hit of results) {
    if (seen.has(hit.conversationId)) continue;
    seen.add(hit.conversationId);
    deduped.push(hit);
    if (deduped.length >= limit) break;
  }

  return deduped;
}
