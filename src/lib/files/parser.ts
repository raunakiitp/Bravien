import { isImageMime } from "@/lib/files/validate";

export interface ParseResult {
  kind: "text" | "image" | "unsupported";
  text?: string;
  metadata?: Record<string, unknown>;
  warning?: string;
}

/**
 * Extract text (or image metadata) from uploaded file bytes.
 * PDF/DOCX require future libraries — stubs return a clear not-configured message.
 */
export async function parseFile(input: {
  buffer: Buffer;
  mimeType: string;
  filename: string;
}): Promise<ParseResult> {
  const { buffer, mimeType, filename } = input;

  if (isImageMime(mimeType)) {
    return {
      kind: "image",
      metadata: {
        filename,
        mimeType,
        size: buffer.byteLength,
        /** Vision models can use the stored file path / data URL separately */
        forVision: true,
      },
    };
  }

  if (
    mimeType === "text/plain" ||
    mimeType === "text/markdown" ||
    mimeType === "text/csv"
  ) {
    return {
      kind: "text",
      text: buffer.toString("utf8"),
      metadata: { filename, mimeType },
    };
  }

  if (mimeType === "application/json") {
    const raw = buffer.toString("utf8");
    try {
      const parsed = JSON.parse(raw) as unknown;
      return {
        kind: "text",
        text: JSON.stringify(parsed, null, 2),
        metadata: { filename, mimeType },
      };
    } catch {
      return {
        kind: "text",
        text: raw,
        metadata: { filename, mimeType, parseWarning: "invalid_json" },
        warning: "File claimed to be JSON but failed to parse; returned raw text.",
      };
    }
  }

  if (mimeType === "application/pdf") {
    return {
      kind: "unsupported",
      warning:
        "PDF parser not fully configured. Install and wire a PDF extraction library to enable text extraction.",
      metadata: { filename, mimeType, size: buffer.byteLength },
    };
  }

  if (
    mimeType ===
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  ) {
    return {
      kind: "unsupported",
      warning:
        "DOCX parser not fully configured. Install and wire a DOCX extraction library to enable text extraction.",
      metadata: { filename, mimeType, size: buffer.byteLength },
    };
  }

  return {
    kind: "unsupported",
    warning: `No parser available for MIME type: ${mimeType}`,
    metadata: { filename, mimeType, size: buffer.byteLength },
  };
}
