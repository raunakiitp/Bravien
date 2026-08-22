import { prisma } from "@/lib/db/prisma";
import type { MemoryDTO, MemoryType } from "@/types";
import type { Memory, MemoryType as PrismaMemoryType } from "@prisma/client";

function toDTO(row: Memory): MemoryDTO {
  return {
    id: row.id,
    userId: row.userId,
    type: row.type as MemoryType,
    content: row.content,
    source: row.sourceConversationId,
    createdAt: row.createdAt.toISOString(),
    updatedAt: row.updatedAt.toISOString(),
  };
}

export interface CreateMemoryInput {
  userId: string;
  type: MemoryType;
  content: string;
  sourceConversationId?: string | null;
}

export interface UpdateMemoryInput {
  content?: string;
  type?: MemoryType;
  sourceConversationId?: string | null;
}

export async function listMemories(
  userId: string,
  options?: { type?: MemoryType; take?: number },
): Promise<MemoryDTO[]> {
  const rows = await prisma.memory.findMany({
    where: {
      userId,
      ...(options?.type ? { type: options.type as PrismaMemoryType } : {}),
    },
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
  const row = await prisma.memory.create({
    data: {
      userId: input.userId,
      type: input.type as PrismaMemoryType,
      content: input.content,
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
      ...(input.content !== undefined ? { content: input.content } : {}),
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

/** Flat string list for system prompt injection. */
export async function getMemoryContentsForPrompt(
  userId: string,
  take = 20,
): Promise<string[]> {
  const rows = await prisma.memory.findMany({
    where: { userId },
    orderBy: { updatedAt: "desc" },
    take,
    select: { content: true },
  });
  return rows.map((r) => r.content);
}
