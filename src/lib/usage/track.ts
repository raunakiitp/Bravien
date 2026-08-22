import { prisma } from "@/lib/db/prisma";

export interface TrackUsageInput {
  userId: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  requestType: string;
  conversationId?: string | null;
}

export async function trackUsage(input: TrackUsageInput) {
  return prisma.usageRecord.create({
    data: {
      userId: input.userId,
      model: input.model,
      inputTokens: Math.max(0, Math.floor(input.inputTokens)),
      outputTokens: Math.max(0, Math.floor(input.outputTokens)),
      requestType: input.requestType,
      conversationId: input.conversationId ?? null,
    },
  });
}
