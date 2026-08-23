import { NextResponse } from "next/server";

import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { getConfig } from "@/lib/config";
import { isDatabaseReachable } from "@/lib/db/available";
import { prisma } from "@/lib/db/prisma";
import { isFeatureEnabled } from "@/lib/features";
import { parseFile } from "@/lib/files/parser";
import { storeFile } from "@/lib/files/storage";
import { ALLOWED_EXTENSIONS, validateUpload } from "@/lib/files/validate";
import { logger } from "@/lib/observability/logger";
import { chunkDocument } from "@/lib/rag/chunker";
import {
  badRequest,
  forbidden,
  jsonError,
  notFound,
  tooManyRequests,
  unauthorized,
} from "@/lib/security/errors";
import { rateLimit } from "@/lib/security/rate-limit";
import type { Prisma } from "@prisma/client";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const RATE_LIMIT = 20;
const RATE_WINDOW_MS = 60_000;
const MAX_FILENAME_LENGTH = 255;

function displayName(raw: string): string {
  const base = raw.split(/[\\/]/).pop() ?? "file";
  const cleaned = base
    .replace(/[\u0000-\u001f\u007f]/g, "")
    .replace(/^\.+/, "")
    .trim();
  return (cleaned || "file").slice(0, MAX_FILENAME_LENGTH);
}

export async function GET(
  _request: Request,
  props: { params: Promise<{ id: string }> },
) {
  const { id: projectId } = await props.params;
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

  const project = await prisma.project.findFirst({
    where: { id: projectId, userId },
    select: { id: true },
  });
  if (!project) return notFound("Project not found.");

  const files = await prisma.attachment.findMany({
    where: { projectId, userId },
    orderBy: { createdAt: "desc" },
    select: {
      id: true,
      filename: true,
      mimeType: true,
      size: true,
      status: true,
      createdAt: true,
      _count: {
        select: { documentChunks: true },
      },
    },
  });

  return NextResponse.json({
    items: files.map((f) => ({
      ...f,
      createdAt: f.createdAt.toISOString(),
      chunkCount: f._count.documentChunks,
    })),
  });
}

export async function POST(
  request: Request,
  props: { params: Promise<{ id: string }> },
) {
  const { id: projectId } = await props.params;

  if (!isFeatureEnabled("file_uploads")) {
    return forbidden("File uploads are disabled.");
  }

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

  // Verify project ownership
  const project = await prisma.project.findFirst({
    where: { id: projectId, userId },
    select: { id: true, name: true },
  });
  if (!project) return notFound("Project not found.");

  const limit = rateLimit(`upload:${userId}`, RATE_LIMIT, RATE_WINDOW_MS);
  if (!limit.allowed) {
    return tooManyRequests("Too many uploads. Try again shortly.");
  }

  const { maxUploadBytes } = getConfig();
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return badRequest("Expected a multipart/form-data body with a `file` field.");
  }

  const entry = form.get("file");
  if (!(entry instanceof File)) {
    return badRequest("Missing `file` field.");
  }

  const filename = displayName(entry.name || "file");
  const mimeType = entry.type || "application/octet-stream";

  const validation = validateUpload({
    mimeType,
    size: entry.size,
    filename,
  });
  if (!validation.ok) {
    return jsonError(
      validation.code === "FILE_TOO_LARGE" ? 413 : 415,
      validation.code ?? "INVALID_FILE",
      validation.error ?? "File rejected.",
    );
  }

  const buffer = Buffer.from(await entry.arrayBuffer());
  if (buffer.byteLength > maxUploadBytes) {
    return jsonError(413, "FILE_TOO_LARGE", "Upload is too large.");
  }

  // Extract text
  const parsed = await parseFile({ buffer, mimeType, filename });
  if (parsed.kind !== "text" || !parsed.text) {
    return jsonError(
      415,
      "UNREADABLE_FILE",
      parsed.warning ??
        `Bravien could not read this file. Supported: ${ALLOWED_EXTENSIONS.join(", ")}.`,
    );
  }

  // Store file safely
  let stored;
  try {
    stored = await storeFile({ buffer, filename, userId });
  } catch (error) {
    logger.error("upload.store_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "STORAGE_ERROR", "Could not save the file.");
  }

  // Chunk text
  const chunks = chunkDocument(parsed.text);

  // Save to database with chunks
  try {
    const attachment = await prisma.attachment.create({
      data: {
        userId,
        projectId,
        filename,
        mimeType,
        size: buffer.byteLength,
        storagePath: stored.storageKey,
        extractedText: parsed.text,
        status: "READY",
        metadata: (parsed.metadata ?? undefined) as Prisma.InputJsonValue | undefined,
        documentChunks: {
          create: chunks.map((c) => ({
            projectId,
            chunkIndex: c.chunkIndex,
            content: c.content,
            tokenCount: c.tokenCount,
            metadata: {
              filename,
              chunkIndex: c.chunkIndex,
            },
          })),
        },
      },
      select: {
        id: true,
        filename: true,
        mimeType: true,
        size: true,
        status: true,
        createdAt: true,
      },
    });

    logger.info("project.file_uploaded", {
      projectId,
      userId,
      filename,
      chunksCount: chunks.length,
    });

    return NextResponse.json(
      {
        id: attachment.id,
        filename: attachment.filename,
        mimeType: attachment.mimeType,
        size: attachment.size,
        status: attachment.status,
        chunkCount: chunks.length,
        createdAt: attachment.createdAt.toISOString(),
      },
      { status: 201 },
    );
  } catch (error) {
    logger.error("project.file_db_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "DB_ERROR", "Could not save file records.");
  }
}
