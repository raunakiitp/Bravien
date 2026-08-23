import { prisma } from "@/lib/db/prisma";
import type { Project, Prisma } from "@prisma/client";

export interface ProjectDTO {
  id: string;
  userId: string;
  name: string;
  description: string | null;
  instructions: string | null;
  createdAt: string;
  updatedAt: string;
  conversationCount?: number;
  attachmentCount?: number;
}

export interface ProjectWithDetailsDTO extends ProjectDTO {
  conversations: Array<{
    id: string;
    title: string;
    model: string;
    updatedAt: string;
  }>;
  attachments: Array<{
    id: string;
    filename: string;
    mimeType: string;
    size: number;
    createdAt: string;
  }>;
  memories: Array<{
    id: string;
    type: string;
    content: string;
    updatedAt: string;
  }>;
}

function toProjectDTO(
  row: Project & {
    _count?: { conversations: number; attachments: number };
  },
): ProjectDTO {
  return {
    id: row.id,
    userId: row.userId,
    name: row.name,
    description: row.description,
    instructions: row.instructions,
    createdAt: row.createdAt.toISOString(),
    updatedAt: row.updatedAt.toISOString(),
    conversationCount: row._count?.conversations,
    attachmentCount: row._count?.attachments,
  };
}

export async function listProjects(userId: string): Promise<ProjectDTO[]> {
  const rows = await prisma.project.findMany({
    where: { userId },
    orderBy: { updatedAt: "desc" },
    include: {
      _count: {
        select: {
          conversations: true,
          attachments: true,
        },
      },
    },
  });
  return rows.map(toProjectDTO);
}

export async function getProject(
  userId: string,
  id: string,
): Promise<ProjectWithDetailsDTO | null> {
  const row = await prisma.project.findFirst({
    where: { id, userId },
    include: {
      _count: {
        select: {
          conversations: true,
          attachments: true,
        },
      },
      conversations: {
        orderBy: { updatedAt: "desc" },
        select: {
          id: true,
          title: true,
          model: true,
          updatedAt: true,
        },
      },
      attachments: {
        orderBy: { createdAt: "desc" },
        select: {
          id: true,
          filename: true,
          mimeType: true,
          size: true,
          createdAt: true,
        },
      },
      memories: {
        orderBy: { updatedAt: "desc" },
        select: {
          id: true,
          type: true,
          content: true,
          updatedAt: true,
        },
      },
    },
  });

  if (!row) return null;

  return {
    ...toProjectDTO(row),
    conversations: row.conversations.map((c) => ({
      ...c,
      updatedAt: c.updatedAt.toISOString(),
    })),
    attachments: row.attachments.map((a) => ({
      ...a,
      createdAt: a.createdAt.toISOString(),
    })),
    memories: row.memories.map((m) => ({
      ...m,
      updatedAt: m.updatedAt.toISOString(),
    })),
  };
}

export async function createProject(input: {
  userId: string;
  name: string;
  description?: string | null;
  instructions?: string | null;
}): Promise<ProjectDTO> {
  const row = await prisma.project.create({
    data: {
      userId: input.userId,
      name: input.name.trim().slice(0, 200),
      description: input.description?.trim() || null,
      instructions: input.instructions?.trim() || null,
    },
  });
  return toProjectDTO(row);
}

export async function updateProject(
  userId: string,
  id: string,
  patch: {
    name?: string;
    description?: string | null;
    instructions?: string | null;
  },
): Promise<ProjectDTO | null> {
  const existing = await prisma.project.findFirst({
    where: { id, userId },
    select: { id: true },
  });
  if (!existing) return null;

  const data: Prisma.ProjectUpdateInput = {};
  if (patch.name !== undefined) data.name = patch.name.trim().slice(0, 200);
  if (patch.description !== undefined)
    data.description = patch.description?.trim() || null;
  if (patch.instructions !== undefined)
    data.instructions = patch.instructions?.trim() || null;

  const updated = await prisma.project.update({
    where: { id },
    data,
  });

  return toProjectDTO(updated);
}

export async function deleteProject(
  userId: string,
  id: string,
): Promise<boolean> {
  const existing = await prisma.project.findFirst({
    where: { id, userId },
    select: { id: true },
  });
  if (!existing) return false;

  await prisma.project.delete({ where: { id } });
  return true;
}
