import { prisma } from "@/lib/db/prisma";

export interface RetrievedChunk {
  chunkId: string;
  attachmentId: string;
  filename: string;
  chunkIndex: number;
  content: string;
  score: number;
}

export interface RetrieveContextOptions {
  userId: string;
  projectId?: string | null;
  query: string;
  limit?: number;
}

/**
 * Clean query into search keywords.
 */
function extractKeywords(query: string): string[] {
  return query
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP_WORDS.has(w));
}

const STOP_WORDS = new Set([
  "the", "and", "is", "in", "it", "of", "to", "this", "that", "with", "for",
  "on", "at", "by", "from", "as", "an", "are", "was", "were", "be", "been",
  "what", "when", "where", "who", "which", "how", "why", "can", "could",
  "will", "would", "should", "tell", "explain", "about", "give", "me", "you",
]);

/**
 * Score a text against search keywords and exact phrases (lexical TF + phrase weighting).
 */
function scoreText(
  text: string,
  keywords: string[],
  filename: string,
  rawQuery: string,
): number {
  if (!keywords.length && !rawQuery.trim()) return 0;
  const lower = text.toLowerCase();
  const lowerName = filename.toLowerCase();
  const cleanQuery = rawQuery.toLowerCase().trim();

  let score = 0;

  // 1. Exact phrase match bonus (+12 points)
  if (cleanQuery.length > 4 && lower.includes(cleanQuery)) {
    score += 12;
  }

  // 2. Filename match bonus (+6 points)
  for (const kw of keywords) {
    if (lowerName.includes(kw)) score += 6;
  }

  // 3. Keyword occurrence scoring with diminishing returns
  for (const kw of keywords) {
    let pos = 0;
    let occurrences = 0;
    while ((pos = lower.indexOf(kw, pos)) !== -1) {
      occurrences += 1;
      pos += kw.length;
      if (occurrences > 8) break;
    }
    score += occurrences * 1.5;
  }

  return score;
}

/**
 * Retrieve relevant document chunks within a project for a user.
 */
export async function retrieveRelevantContext(
  options: RetrieveContextOptions,
): Promise<RetrievedChunk[]> {
  const { userId, projectId, query } = options;
  const limit = Math.min(options.limit ?? 4, 8);
  const keywords = extractKeywords(query);

  if (!projectId || !query.trim()) return [];

  // Verify project ownership strictly
  const project = await prisma.project.findFirst({
    where: { id: projectId, userId },
    select: { id: true },
  });
  if (!project) return [];

  // Fetch chunks belonging to this project and user's attachments
  const chunks = await prisma.documentChunk.findMany({
    where: {
      projectId,
      attachment: { userId },
    },
    include: {
      attachment: {
        select: { id: true, filename: true },
      },
    },
    take: 120,
  });

  if (!chunks.length) {
    // If no chunks yet but attachment has extractedText, query directly
    const attachments = await prisma.attachment.findMany({
      where: { projectId, userId, status: "READY" },
      select: { id: true, filename: true, extractedText: true },
      take: 10,
    });

    const fallbackHits: RetrievedChunk[] = [];
    for (const att of attachments) {
      if (!att.extractedText) continue;
      const score = scoreText(att.extractedText, keywords, att.filename, query);
      if (score > 0 || keywords.length === 0) {
        fallbackHits.push({
          chunkId: att.id,
          attachmentId: att.id,
          filename: att.filename,
          chunkIndex: 0,
          content: att.extractedText.slice(0, 1000),
          score,
        });
      }
    }
    return fallbackHits.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  // Score chunks
  const scored: RetrievedChunk[] = chunks.map((chunk) => ({
    chunkId: chunk.id,
    attachmentId: chunk.attachmentId,
    filename: chunk.attachment.filename,
    chunkIndex: chunk.chunkIndex,
    content: chunk.content,
    score: scoreText(chunk.content, keywords, chunk.attachment.filename, query),
  }));

  // Filter chunks with positive relevance score
  const filtered = keywords.length > 0 ? scored.filter((c) => c.score > 0) : scored;
  const sorted = filtered.sort((a, b) => b.score - a.score);

  // Deduplicate chunks (avoid excessive adjacent duplicates from same file)
  const seenFiles = new Map<string, number>();
  const deduped: RetrievedChunk[] = [];

  for (const chunk of sorted) {
    const count = seenFiles.get(chunk.filename) ?? 0;
    if (count < 2) {
      // Max 2 chunks per file for variety across project docs
      seenFiles.set(chunk.filename, count + 1);
      deduped.push(chunk);
      if (deduped.length >= limit) break;
    }
  }

  return deduped;
}

/**
 * Format retrieved chunks into a prompt-ready context block.
 */
export function formatRetrievedContext(chunks: RetrievedChunk[]): string {
  if (!chunks.length) return "";

  const formatted = chunks.map(
    (chunk) =>
      `[Source Document: ${chunk.filename} (Section ${chunk.chunkIndex + 1})]\n${chunk.content.slice(0, 1000)}`,
  );

  return `Relevant project reference documents:\n\n${formatted.join("\n\n---\n\n")}\n\nInstructions for reference documents:\n- Use the above document excerpts to answer the question.\n- Include source references (e.g. "[${chunks[0].filename}]") when citing details.\n- If the documents do not contain enough information to answer the question, explicitly state: "The uploaded project documents do not contain sufficient information to answer this question."`;
}
