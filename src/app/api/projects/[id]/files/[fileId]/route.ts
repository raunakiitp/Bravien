import { NextResponse } from "next/server";

import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { prisma } from "@/lib/db/prisma";
import { deleteStoredFile } from "@/lib/files/storage";
import { logger } from "@/lib/observability/logger";
import { jsonError, notFound, unauthorized } from "@/lib/security/errors";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function DELETE(
  _request: Request,
  props: { params: Promise<{ id: string; fileId: string }> },
) {
  const { id: projectId, fileId } = await props.params;

  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(503, "PERSISTENCE_UNAVAILABLE", "Database unavailable.");
  }

  const attachment = await prisma.attachment.findFirst({
    where: { id: fileId, projectId, userId },
    select: { id: true, storagePath: true },
  });

  if (!attachment) {
    return notFound("File not found.");
  }

  try {
    await prisma.attachment.delete({
      where: { id: fileId },
    });

    if (attachment.storagePath) {
      await deleteStoredFile(attachment.storagePath).catch(() => undefined);
    }

    return new NextResponse(null, { status: 204 });
  } catch (error) {
    logger.error("project.file_delete_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "FILE_DELETE_ERROR", "Could not delete file.");
  }
}
