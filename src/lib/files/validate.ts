import { getConfig } from "@/lib/config";

/**
 * File types Bravien can actually read.
 *
 * The list is deliberately short: it contains exactly the formats
 * `parseFile` extracts text from today. Images are absent because this
 * architecture has no vision tower — accepting a PNG would store bytes the model
 * can never see. PDF and DOCX are absent because no extractor is wired yet.
 * Rejecting them with a clear message beats accepting them and attaching nothing
 * (§72).
 */
export const ALLOWED_MIME_TYPES = new Set([
  "text/plain",
  "text/markdown",
  "text/csv",
  "application/json",
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]);

/** Extensions matching the allowlist, for a friendlier error message. */
export const ALLOWED_EXTENSIONS = [".txt", ".md", ".csv", ".json", ".pdf", ".docx"];

export interface FileValidationResult {
  ok: boolean;
  error?: string;
  code?: string;
}

export const MAX_FILENAME_LENGTH = 255;

/**
 * Sanitizes user-provided filenames to prevent path traversal and shell exploits.
 */
export function sanitizeFilename(raw: string): string {
  if (!raw) return "file";
  // Strip path traversal sequences and directory separators
  const base = raw
    .replace(/\\/g, "/")
    .split("/")
    .pop() ?? "file";

  // Remove null bytes, control characters, leading dots
  const cleaned = base
    .replace(/[\u0000-\u001f\u007f]/g, "")
    .replace(/^\.+/, "")
    .replace(/[<>:"/\\|?*]/g, "_")
    .trim();

  return (cleaned || "file").slice(0, MAX_FILENAME_LENGTH);
}

export function validateUpload(file: {
  mimeType: string;
  size: number;
  filename?: string;
}): FileValidationResult {
  const { maxUploadBytes } = getConfig();

  if (file.filename) {
    const clean = sanitizeFilename(file.filename);
    const hasValidExt = ALLOWED_EXTENSIONS.some((ext) =>
      clean.toLowerCase().endsWith(ext),
    );
    if (!hasValidExt) {
      return {
        ok: false,
        code: "UNSUPPORTED_EXTENSION",
        error: `File extension is not supported. Supported: ${ALLOWED_EXTENSIONS.join(", ")}.`,
      };
    }
  }

  if (!file.mimeType || !ALLOWED_MIME_TYPES.has(file.mimeType)) {
    return {
      ok: false,
      code: "UNSUPPORTED_MIME",
      error:
        `Bravien cannot read ${file.mimeType || "that file type"}. ` +
        `Supported: ${ALLOWED_EXTENSIONS.join(", ")}.`,
    };
  }

  if (!Number.isFinite(file.size) || file.size < 0) {
    return {
      ok: false,
      code: "INVALID_SIZE",
      error: "Invalid file size",
    };
  }

  if (file.size === 0) {
    return { ok: false, code: "EMPTY_FILE", error: "File is empty." };
  }

  if (file.size > maxUploadBytes) {
    return {
      ok: false,
      code: "FILE_TOO_LARGE",
      error: `File exceeds the maximum size of ${Math.floor(maxUploadBytes / 1024 / 1024)} MB.`,
    };
  }

  return { ok: true };
}

export function isImageMime(mimeType: string): boolean {
  return mimeType.startsWith("image/");
}
