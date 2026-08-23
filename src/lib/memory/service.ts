import { prisma } from "@/lib/db/prisma";
import type { MemoryDTO, MemoryType } from "@/types";
import type { Memory, MemoryType as PrismaMemoryType } from "@prisma/client";

function toDTO(row: Memory): MemoryDTO {
  return {
    id: row.id,
    userId: row.userId,
    projectId: row.projectId ?? null,
    type: row.type as MemoryType,
    content: row.content,
    source: row.sourceConversationId,
    createdAt: row.createdAt.toISOString(),
    updatedAt: row.updatedAt.toISOString(),
  };
}

export interface CreateMemoryInput {
  userId: string;
  projectId?: string | null;
  type: MemoryType;
  content: string;
  sourceConversationId?: string | null;
}

export interface UpdateMemoryInput {
  content?: string;
  projectId?: string | null;
  type?: MemoryType;
  sourceConversationId?: string | null;
}

export async function listMemories(
  userId: string,
  options?: { type?: MemoryType; projectId?: string | null; take?: number },
): Promise<MemoryDTO[]> {
  const where: Record<string, unknown> = { userId };
  if (options?.type) where.type = options.type as PrismaMemoryType;
  if (options?.projectId !== undefined) where.projectId = options.projectId;

  const rows = await prisma.memory.findMany({
    where,
    orderBy: { updatedAt: "desc" },
    take: options?.take ?? 100,
  });
  return rows.map(toDTO);
}

export async function getMemory(
  userId: string,
  id: string,
): Promise<MemoryDTO | null> {
  const row = await prisma.memory.findFirst({ where: { id, userId } });
  return row ? toDTO(row) : null;
}

export async function createMemory(
  input: CreateMemoryInput,
): Promise<MemoryDTO> {
  const normalizedContent = input.content.trim();

  // Duplicate prevention: check if identical memory already exists for user/project
  const existing = await prisma.memory.findFirst({
    where: {
      userId: input.userId,
      projectId: input.projectId ?? null,
      type: input.type as PrismaMemoryType,
      content: normalizedContent,
    },
  });

  if (existing) {
    return toDTO(existing);
  }

  const row = await prisma.memory.create({
    data: {
      userId: input.userId,
      projectId: input.projectId ?? null,
      type: input.type as PrismaMemoryType,
      content: normalizedContent,
      sourceConversationId: input.sourceConversationId ?? null,
    },
  });
  return toDTO(row);
}

export async function updateMemory(
  userId: string,
  id: string,
  input: UpdateMemoryInput,
): Promise<MemoryDTO | null> {
  const existing = await prisma.memory.findFirst({ where: { id, userId } });
  if (!existing) return null;

  const row = await prisma.memory.update({
    where: { id },
    data: {
      ...(input.content !== undefined ? { content: input.content.trim() } : {}),
      ...(input.projectId !== undefined ? { projectId: input.projectId } : {}),
      ...(input.type !== undefined
        ? { type: input.type as PrismaMemoryType }
        : {}),
      ...(input.sourceConversationId !== undefined
        ? { sourceConversationId: input.sourceConversationId }
        : {}),
    },
  });
  return toDTO(row);
}

export async function deleteMemory(
  userId: string,
  id: string,
): Promise<boolean> {
  const existing = await prisma.memory.findFirst({ where: { id, userId } });
  if (!existing) return false;
  await prisma.memory.delete({ where: { id } });
  return true;
}

export async function clearMemories(
  userId: string,
  options?: { projectId?: string | null; type?: MemoryType },
): Promise<number> {
  const where: Record<string, unknown> = { userId };
  if (options?.projectId !== undefined) {
    where.projectId = options.projectId;
  }
  if (options?.type) {
    where.type = options.type as PrismaMemoryType;
  }
  const result = await prisma.memory.deleteMany({ where });
  return result.count;
}

export async function searchMemories(
  userId: string,
  query: string,
  options?: { projectId?: string | null; type?: MemoryType; limit?: number },
): Promise<MemoryDTO[]> {
  const q = query.trim();
  if (!q) return listMemories(userId, options);

  const where: Record<string, unknown> = {
    userId,
    content: { contains: q, mode: "insensitive" },
  };
  if (options?.projectId !== undefined) {
    where.projectId = options.projectId;
  }
  if (options?.type) {
    where.type = options.type as PrismaMemoryType;
  }

  const rows = await prisma.memory.findMany({
    where,
    orderBy: { updatedAt: "desc" },
    take: options?.limit ?? 20,
  });
  return rows.map(toDTO);
}

/**
 * Retrieve relevance-ranked memories for a specific user query.
 * Keywords and phrases are scored to inject the most pertinent facts first.
 */
export async function getRelevantMemoriesForQuery(
  userId: string,
  query: string,
  options?: { projectId?: string | null; limit?: number },
): Promise<string[]> {
  const limit = options?.limit ?? 8;
  const q = query.toLowerCase().trim();
  const keywords = q
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);

  const where: Record<string, unknown> = { userId };
  if (options?.projectId) {
    where.OR = [{ projectId: options.projectId }, { projectId: null }];
  }

  const rows = await prisma.memory.findMany({
    where,
    orderBy: { updatedAt: "desc" },
    take: 50,
    select: { content: true, type: true },
  });

  if (!rows.length) return [];

  // Score memories based on keyword relevance and category priority
  const scored = rows.map((row) => {
    const lowerContent = row.content.toLowerCase();
    let score = 0;

    // Direct phrase match bonus
    if (q.length > 3 && lowerContent.includes(q)) score += 10;

    // Keyword match
    for (const kw of keywords) {
      if (lowerContent.includes(kw)) score += 3;
    }

    // Category priority weights
    if (row.type === "PREFERENCE" || row.type === "INSTRUCTION") score += 2;
    if (row.type === "FACT" || row.type === "PROFILE") score += 1.5;

    return { content: row.content, score };
  });

  const sorted = scored.sort((a, b) => b.score - a.score);
  return sorted.slice(0, limit).map((s) => s.content);
}

/** Flat string list for system prompt injection, user and project-scoped. */
export async function getMemoryContentsForPrompt(
  userId: string,
  options?: { projectId?: string | null; query?: string; take?: number },
): Promise<string[]> {
  if (options?.query?.trim()) {
    return getRelevantMemoriesForQuery(userId, options.query, {
      projectId: options.projectId,
      limit: options.take ?? 10,
    });
  }

  const take = options?.take ?? 20;
  const where: Record<string, unknown> = {
    userId,
  };

  if (options?.projectId) {
    where.OR = [
      { projectId: options.projectId },
      { projectId: null },
    ];
  }

  const rows = await prisma.memory.findMany({
    where,
    orderBy: { updatedAt: "desc" },
    take,
    select: { content: true },
  });
  return rows.map((r) => r.content);
}
