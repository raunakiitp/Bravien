export interface DocumentChunkSpec {
  chunkIndex: number;
  content: string;
  tokenCount: number;
}

export interface ChunkOptions {
  chunkSize?: number;
  chunkOverlap?: number;
}

const DEFAULT_CHUNK_SIZE = 1200; // ~300 tokens
const DEFAULT_CHUNK_OVERLAP = 200; // ~50 tokens

/**
 * Split text into overlapping chunks, preferring paragraph and sentence boundaries.
 */
export function chunkDocument(
  text: string,
  options: ChunkOptions = {},
): DocumentChunkSpec[] {
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
  const chunkOverlap = options.chunkOverlap ?? DEFAULT_CHUNK_OVERLAP;

  const normalized = text.replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];

  if (normalized.length <= chunkSize) {
    return [
      {
        chunkIndex: 0,
        content: normalized,
        tokenCount: Math.ceil(normalized.length / 4),
      },
    ];
  }

  const chunks: DocumentChunkSpec[] = [];
  let startIndex = 0;
  let chunkIndex = 0;

  while (startIndex < normalized.length) {
    let endIndex = startIndex + chunkSize;

    if (endIndex < normalized.length) {
      // Look for natural boundary near endIndex: paragraph > sentence > space
      const searchWindow = normalized.slice(
        Math.max(startIndex, endIndex - 150),
        Math.min(normalized.length, endIndex + 50),
      );

      const paragraphBreak = searchWindow.lastIndexOf("\n\n");
      const sentenceBreak = searchWindow.search(/[.!?]\s+[A-Z0-9]/);
      const lineBreak = searchWindow.lastIndexOf("\n");
      const spaceBreak = searchWindow.lastIndexOf(" ");

      const windowStart = Math.max(startIndex, endIndex - 150);

      if (paragraphBreak !== -1 && windowStart + paragraphBreak > startIndex + 100) {
        endIndex = windowStart + paragraphBreak + 2;
      } else if (sentenceBreak !== -1 && windowStart + sentenceBreak > startIndex + 100) {
        endIndex = windowStart + sentenceBreak + 2;
      } else if (lineBreak !== -1 && windowStart + lineBreak > startIndex + 100) {
        endIndex = windowStart + lineBreak + 1;
      } else if (spaceBreak !== -1 && windowStart + spaceBreak > startIndex + 100) {
        endIndex = windowStart + spaceBreak + 1;
      }
    } else {
      endIndex = normalized.length;
    }

    const chunkText = normalized.slice(startIndex, endIndex).trim();
    if (chunkText.length > 0) {
      chunks.push({
        chunkIndex,
        content: chunkText,
        tokenCount: Math.ceil(chunkText.length / 4),
      });
      chunkIndex += 1;
    }

    if (endIndex >= normalized.length) break;

    // Advance with overlap
    startIndex = Math.max(startIndex + 1, endIndex - chunkOverlap);
  }

  return chunks;
}
