/**
 * POST /api/files — attach a text document to a conversation.
 *
 * Uploaded bytes are never executed, never interpreted as code, and never given a
 * caller-controlled path: `storeFile` writes them under `uploads/` with a random
 * name, and the original filename is only ever data (§53, §54).
 *
 * The extracted text is what the model sees, and it is passed as untrusted
 * content — the system prompt already instructs Bravien not to obey instructions
 * found inside documents.
 */

import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { getConfig } from "@/lib/config";
import { isDatabaseReachable } from "@/lib/db/available";
import { isFeatureEnabled } from "@/lib/features";
import { parseFile } from "@/lib/files/parser";
import { storeFile } from "@/lib/files/storage";
import { ALLOWED_EXTENSIONS, sanitizeFilename, validateUpload } from "@/lib/files/validate";
import { logger } from "@/lib/observability/logger";
import {
  badRequest,
  forbidden,
  jsonError,
  tooManyRequests,
  unauthorized,
} from "@/lib/security/errors";
import { rateLimit } from "@/lib/security/rate-limit";
import type { Prisma } from "@prisma/client";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const RATE_LIMIT = 20;
const RATE_WINDOW_MS = 60_000;

export async function POST(request: Request) {
  if (!isFeatureEnabled("file_uploads")) {
    return forbidden("File uploads are disabled (FEATURE_FILE_UPLOADS).");
  }

  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  const limit = rateLimit(`upload:${userId}`, RATE_LIMIT, RATE_WINDOW_MS);
  if (!limit.allowed) {
    return tooManyRequests("Too many uploads. Try again shortly.");
  }

  const { maxUploadBytes } = getConfig();
  const declared = Number(request.headers.get("content-length") ?? 0);
  // Reject from the header before buffering: the point of a size limit is not
  // reading the bytes in the first place.
  if (Number.isFinite(declared) && declared > maxUploadBytes * 1.1) {
    return jsonError(413, "FILE_TOO_LARGE", "Upload is too large.");
  }

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

  const filename = sanitizeFilename(entry.name || "file");
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
  // The declared size can lie; the real byte count is authoritative.
  if (buffer.byteLength > maxUploadBytes) {
    return jsonError(413, "FILE_TOO_LARGE", "Upload is too large.");
  }

  const parsed = await parseFile({ buffer, mimeType, filename });
  if (parsed.kind !== "text" || !parsed.text) {
    return jsonError(
      415,
      "UNREADABLE_FILE",
      parsed.warning ??
        `Bravien could not read this file. Supported: ${ALLOWED_EXTENSIONS.join(", ")}.`,
    );
  }

  let stored;
  try {
    stored = await storeFile({ buffer, filename, userId });
  } catch (error) {
    logger.error("upload.store_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "STORAGE_ERROR", "Could not save the file.");
  }

  // Record it when there is a database; without one the extracted text still
  // comes back and can be attached to the next message.
  let id: string | null = null;
  if (await isDatabaseReachable()) {
    try {
      const { prisma } = await import("@/lib/db/prisma");
      const row = await prisma.attachment.create({
        data: {
          userId,
          filename,
          mimeType,
          size: buffer.byteLength,
          storagePath: stored.storageKey,
          extractedText: parsed.text,
          status: "READY",
          metadata: (parsed.metadata ?? undefined) as
            | Prisma.InputJsonValue
            | undefined,
        },
        select: { id: true },
      });
      id = row.id;
    } catch (error) {
      logger.error("upload.record_failed", {
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return Response.json(
    {
      id,
      filename,
      mimeType,
      size: buffer.byteLength,
      characters: parsed.text.length,
      text: parsed.text,
      ...(parsed.warning ? { warning: parsed.warning } : {}),
    },
    { status: 201 },
  );
}
