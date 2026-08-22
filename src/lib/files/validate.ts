import { getConfig } from "@/lib/config";

export const ALLOWED_MIME_TYPES = new Set([
  "text/plain",
  "text/markdown",
  "text/csv",
  "application/json",
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
]);

export interface FileValidationResult {
  ok: boolean;
  error?: string;
  code?: string;
}

export function validateUpload(file: {
  mimeType: string;
  size: number;
  filename?: string;
}): FileValidationResult {
  const { maxUploadBytes } = getConfig();

  if (!file.mimeType || !ALLOWED_MIME_TYPES.has(file.mimeType)) {
    return {
      ok: false,
      code: "UNSUPPORTED_MIME",
      error: `Unsupported file type: ${file.mimeType || "unknown"}`,
    };
  }

  if (!Number.isFinite(file.size) || file.size < 0) {
    return {
      ok: false,
      code: "INVALID_SIZE",
      error: "Invalid file size",
    };
  }

  if (file.size > maxUploadBytes) {
    return {
      ok: false,
      code: "FILE_TOO_LARGE",
      error: `File exceeds maximum size of ${maxUploadBytes} bytes`,
    };
  }

  return { ok: true };
}

export function isImageMime(mimeType: string): boolean {
  return mimeType.startsWith("image/");
}
