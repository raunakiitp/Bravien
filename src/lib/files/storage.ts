import { randomBytes } from "node:crypto";
import { mkdir, writeFile, readFile, unlink } from "node:fs/promises";
import path from "node:path";
import { getConfig } from "@/lib/config";

function sanitizeExtension(filename: string): string {
  const ext = path.extname(filename).toLowerCase().replace(/[^a-z0-9.]/g, "");
  if (!ext || ext.length > 12) return "";
  return ext;
}

export interface StoredFile {
  storageKey: string;
  absolutePath: string;
  relativePath: string;
  originalFilename: string;
  size: number;
}

function uploadsRoot(): string {
  const { uploadDir } = getConfig();
  return path.isAbsolute(uploadDir)
    ? uploadDir
    : path.join(process.cwd(), uploadDir);
}

export async function ensureUploadsDir(): Promise<string> {
  const root = uploadsRoot();
  await mkdir(root, { recursive: true });
  return root;
}

/**
 * Persist a file under uploads/ with a cryptographically random name.
 */
export async function storeFile(input: {
  buffer: Buffer;
  filename: string;
  userId?: string;
}): Promise<StoredFile> {
  const root = await ensureUploadsDir();
  const ext = sanitizeExtension(input.filename);
  const randomName = randomBytes(24).toString("hex") + ext;
  const relativePath = input.userId
    ? path.join(input.userId, randomName)
    : randomName;
  const absolutePath = path.join(root, relativePath);

  await mkdir(path.dirname(absolutePath), { recursive: true });
  await writeFile(absolutePath, input.buffer);

  return {
    storageKey: relativePath.replace(/\\/g, "/"),
    absolutePath,
    relativePath: relativePath.replace(/\\/g, "/"),
    originalFilename: input.filename,
    size: input.buffer.byteLength,
  };
}

export async function readStoredFile(storageKey: string): Promise<Buffer> {
  const root = uploadsRoot();
  const resolved = path.resolve(root, storageKey);
  if (!resolved.startsWith(path.resolve(root))) {
    throw new Error("Invalid storage key");
  }
  return readFile(resolved);
}

export async function deleteStoredFile(storageKey: string): Promise<void> {
  const root = uploadsRoot();
  const resolved = path.resolve(root, storageKey);
  if (!resolved.startsWith(path.resolve(root))) {
    throw new Error("Invalid storage key");
  }
  await unlink(resolved).catch(() => undefined);
}
