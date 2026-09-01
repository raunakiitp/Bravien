/**
 * Model Response Cache & Deduplication for Bravien.
 *
 * Lightweight, in-memory LRU cache to prevent redundant local model inference
 * for identical safe requests within a project/user context.
 */

import { createHash } from "crypto";

export interface CacheEntry {
  key: string;
  response: string;
  userId: string;
  projectId: string | null;
  createdAt: number;
  expiresAt: number;
}

export interface CacheKeyOptions {
  userId: string;
  projectId?: string | null;
  query: string;
  contextHash?: string;
  profile?: string;
}

export class InferenceCache {
  private static instance: InferenceCache | null = null;
  private cache: Map<string, CacheEntry> = new Map();
  private maxEntries: number = 500;
  private defaultTtlMs: number = 5 * 60 * 1000; // 5 minutes
  private hits: number = 0;
  private misses: number = 0;

  private constructor() {}

  public static getInstance(): InferenceCache {
    if (!InferenceCache.instance) {
      InferenceCache.instance = new InferenceCache();
    }
    return InferenceCache.instance;
  }

  /**
   * Generates a collision-resistant deterministic fingerprint key.
   */
  public generateKey(options: CacheKeyOptions): string {
    const normUser = options.userId.trim();
    const normProject = (options.projectId ?? "global").trim();
    const normQuery = options.query.trim().toLowerCase().replace(/\s+/g, " ");
    const normProfile = options.profile ?? "BALANCED";
    const context = options.contextHash ?? "";

    const raw = `${normUser}:${normProject}:${normProfile}:${normQuery}:${context}`;
    return createHash("sha256").update(raw).digest("hex");
  }

  /**
   * Whether a query is safe to cache.
   * Excludes time-sensitive queries, web searches, destructive actions, and sensitive credentials.
   */
  public isCacheable(query: string, options: { requiresWeb?: boolean; isAction?: boolean } = {}): boolean {
    if (options.requiresWeb || options.isAction) return false;

    const lower = query.toLowerCase();
    // Exclude time/date queries (they change constantly)
    if (/\b(?:time|date|today|now|current|yesterday|tomorrow)\b/i.test(lower)) {
      return false;
    }

    // Exclude destructive operations or secret handling
    if (/\b(?:delete|drop|remove|destroy|truncate|password|token|secret|key|api_key|apikey)\b/i.test(lower)) {
      return false;
    }

    return true;
  }

  /**
   * Retrieves a cached response if valid and not expired.
   */
  public get(key: string, userId: string): string | null {
    const entry = this.cache.get(key);
    if (!entry) {
      this.misses++;
      return null;
    }

    // Security check: Never serve cache entry across different users
    if (entry.userId !== userId) {
      this.misses++;
      return null;
    }

    // Expiry check
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      this.misses++;
      return null;
    }

    // Refresh LRU order
    this.cache.delete(key);
    this.cache.set(key, entry);
    this.hits++;
    return entry.response;
  }

  /**
   * Stores a response in the cache.
   */
  public set(
    key: string,
    response: string,
    options: { userId: string; projectId?: string | null; ttlMs?: number },
  ): void {
    if (!response || !options.userId) return;

    // Evict oldest entry if at capacity
    if (this.cache.size >= this.maxEntries) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }

    const now = Date.now();
    const expiresAt = now + (options.ttlMs ?? this.defaultTtlMs);

    this.cache.set(key, {
      key,
      response,
      userId: options.userId,
      projectId: options.projectId ?? null,
      createdAt: now,
      expiresAt,
    });
  }

  /**
   * Invalidate entries for a user or project.
   */
  public invalidateUser(userId: string): void {
    for (const [key, entry] of this.cache.entries()) {
      if (entry.userId === userId) {
        this.cache.delete(key);
      }
    }
  }

  public clear(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  public size(): number {
    return this.cache.size;
  }

  public getStats(): { hits: number; misses: number; size: number; hitRate: number } {
    const total = this.hits + this.misses;
    const hitRate = total > 0 ? round(this.hits / total, 3) : 0;
    return {
      hits: this.hits,
      misses: this.misses,
      size: this.cache.size,
      hitRate,
    };
  }
}

function round(val: number, digits: number): number {
  const factor = Math.pow(10, digits);
  return Math.round(val * factor) / factor;
}

export const inferenceCache = InferenceCache.getInstance();
