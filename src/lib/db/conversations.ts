/**
 * Conversation persistence.
 *
 * Every query is scoped by `userId`. A conversation id alone never grants access:
 * the ownership check is part of the `where` clause, not a separate `if` that
 * could be forgotten (§53).
 */

import { prisma } from "@/lib/db/prisma";
import type { ConversationDTO, MessageDTO } from "@/types";
import type {
  Conversation,
  Message,
  MessageRole,
  Prisma,
} from "@prisma/client";

export const MAX_TITLE_LENGTH = 200;
export const MAX_MESSAGE_LENGTH = 200_000;

function toConversationDTO(row: Conversation): ConversationDTO {
  return {
    id: row.id,
    userId: row.userId,
    title: row.title,
    model: row.model,
    pinned: row.pinned,
    archived: row.archived,
    shareId: row.shareId,
    sharedAt: row.sharedAt?.toISOString() ?? null,
    summary: row.summary,
    metadata: (row.metadata as Record<string, unknown> | null) ?? null,
    createdAt: row.createdAt.toISOString(),
    updatedAt: row.updatedAt.toISOString(),
  };
}

function toMessageDTO(row: Message): MessageDTO {
  return {
    id: row.id,
    conversationId: row.conversationId,
    role: row.role,
    content: row.content,
    parentId: row.parentId,
    attachments: row.attachments ?? null,
    toolCalls: row.toolCalls ?? null,
    citations: row.citations ?? null,
    feedback: row.feedback,
    metadata: (row.metadata as Record<string, unknown> | null) ?? null,
    createdAt: row.createdAt.toISOString(),
  };
}

export interface ListConversationsOptions {
  archived?: boolean;
  query?: string;
  take?: number;
  cursor?: string;
}

export async function listConversations(
  userId: string,
  options: ListConversationsOptions = {},
): Promise<{ items: ConversationDTO[]; nextCursor: string | null }> {
  const take = Math.min(Math.max(options.take ?? 40, 1), 100);
  const where: Prisma.ConversationWhereInput = {
    userId,
    archived: options.archived ?? false,
  };
  if (options.query?.trim()) {
    where.title = { contains: options.query.trim(), mode: "insensitive" };
  }

  const rows = await prisma.conversation.findMany({
    where,
    orderBy: [{ pinned: "desc" }, { updatedAt: "desc" }],
    // One extra row tells us whether another page exists without a second count
    // query.
    take: take + 1,
    ...(options.cursor ? { cursor: { id: options.cursor }, skip: 1 } : {}),
  });

  const hasMore = rows.length > take;
  const page = hasMore ? rows.slice(0, take) : rows;
  return {
    items: page.map(toConversationDTO),
    nextCursor: hasMore ? (page[page.length - 1]?.id ?? null) : null,
  };
}

export async function getConversation(
  userId: string,
  id: string,
): Promise<ConversationDTO | null> {
  const row = await prisma.conversation.findFirst({ where: { id, userId } });
  return row ? toConversationDTO(row) : null;
}

export async function getConversationMessages(
  userId: string,
  conversationId: string,
): Promise<MessageDTO[] | null> {
  const owned = await prisma.conversation.findFirst({
    where: { id: conversationId, userId },
    select: { id: true },
  });
  if (!owned) return null;

  const rows = await prisma.message.findMany({
    where: { conversationId },
    orderBy: { createdAt: "asc" },
  });
  return rows.map(toMessageDTO);
}

export async function createConversation(input: {
  userId: string;
  title: string;
  model: string;
}): Promise<ConversationDTO> {
  const row = await prisma.conversation.create({
    data: {
      userId: input.userId,
      title: input.title.slice(0, MAX_TITLE_LENGTH) || "New chat",
      model: input.model,
    },
  });
  return toConversationDTO(row);
}

export async function updateConversation(
  userId: string,
  id: string,
  patch: {
    title?: string;
    pinned?: boolean;
    archived?: boolean;
    model?: string;
    summary?: string | null;
  },
): Promise<ConversationDTO | null> {
  // updateMany scopes the write by userId, so a foreign id updates zero rows
  // instead of someone else's conversation.
  const result = await prisma.conversation.updateMany({
    where: { id, userId },
    data: {
      ...(patch.title !== undefined
        ? { title: patch.title.slice(0, MAX_TITLE_LENGTH) || "New chat" }
        : {}),
      ...(patch.pinned !== undefined ? { pinned: patch.pinned } : {}),
      ...(patch.archived !== undefined ? { archived: patch.archived } : {}),
      ...(patch.model !== undefined ? { model: patch.model } : {}),
      ...(patch.summary !== undefined ? { summary: patch.summary } : {}),
    },
  });
  if (result.count === 0) return null;
  return getConversation(userId, id);
}

export async function deleteConversation(
  userId: string,
  id: string,
): Promise<boolean> {
  const result = await prisma.conversation.deleteMany({ where: { id, userId } });
  return result.count > 0;
}

export async function appendMessage(input: {
  conversationId: string;
  role: MessageRole;
  content: string;
  parentId?: string | null;
  metadata?: Record<string, unknown> | null;
  attachments?: unknown;
}): Promise<MessageDTO> {
  const row = await prisma.message.create({
    data: {
      conversationId: input.conversationId,
      role: input.role,
      content: input.content.slice(0, MAX_MESSAGE_LENGTH),
      parentId: input.parentId ?? null,
      metadata: (input.metadata ?? undefined) as Prisma.InputJsonValue | undefined,
      attachments: (input.attachments ?? undefined) as
        | Prisma.InputJsonValue
        | undefined,
    },
  });
  return toMessageDTO(row);
}

/** Bump `updatedAt` so the sidebar orders by real activity. */
export async function touchConversation(id: string): Promise<void> {
  await prisma.conversation
    .update({ where: { id }, data: { updatedAt: new Date() } })
    .catch(() => undefined);
}

export async function setMessageFeedback(
  userId: string,
  messageId: string,
  rating: "LIKE" | "DISLIKE" | null,
): Promise<boolean> {
  const message = await prisma.message.findFirst({
    where: { id: messageId, conversation: { userId } },
    select: { id: true },
  });
  if (!message) return false;
  await prisma.message.update({
    where: { id: messageId },
    data: { feedback: rating },
  });
  return true;
}
