/**
 * Bravien Strategy Memory (§Phase 14).
 *
 * Persists compact, successful execution patterns in a bounded memory cache
 * (max 500 records) to provide advisory guidance on future complex turns.
 */

import { createHash } from "crypto";
import type { StrategyRecord } from "./types";

const MAX_STRATEGY_RECORDS = 500;

export class StrategyMemoryStore {
  private records: StrategyRecord[] = [];

  /**
   * Generates a normalized SHA-256 query fingerprint without storing sensitive tokens.
   */
  public generateFingerprint(query: string): string {
    const normalized = query.trim().toLowerCase().replace(/\s+/g, " ");
    return createHash("sha256").update(normalized).digest("hex").slice(0, 32);
  }

  /**
   * Records a successful strategy pattern with secret sanitization and bounded size limits.
   */
  public recordStrategy(record: StrategyRecord): boolean {
    // Secret filtering: Do not record strategies for queries containing secrets
    if (this.containsSensitiveData(record.queryFingerprint)) {
      return false;
    }

    // Deduplication check
    const existingIndex = this.records.findIndex(
      (r) =>
        r.queryFingerprint === record.queryFingerprint &&
        r.userId === record.userId &&
        r.projectId === record.projectId,
    );

    if (existingIndex !== -1) {
      this.records[existingIndex] = { ...record, timestamp: Date.now() };
      return true;
    }

    // FIFO eviction when reaching max capacity
    if (this.records.length >= MAX_STRATEGY_RECORDS) {
      this.records.shift();
    }

    this.records.push({ ...record, timestamp: Date.now() });
    return true;
  }

  /**
   * Looks up a previously successful strategy for a query fingerprint and tenant scope.
   */
  public lookupStrategy(
    fingerprint: string,
    options?: { userId?: string; projectId?: string },
  ): StrategyRecord | null {
    const match = this.records.find((r) => {
      if (r.queryFingerprint !== fingerprint) return false;
      if (options?.userId && r.userId && r.userId !== options.userId) return false;
      if (options?.projectId && r.projectId && r.projectId !== options.projectId) return false;
      return true;
    });

    return match ? { ...match } : null;
  }

  /**
   * Returns current count of stored strategy records.
   */
  public getCount(userId?: string): number {
    if (!userId) return this.records.length;
    return this.records.filter((r) => r.userId === userId).length;
  }

  /**
   * Clears the strategy memory store.
   */
  public clear(): void {
    this.records = [];
  }

  private containsSensitiveData(text: string): boolean {
    const lower = text.toLowerCase();
    return (
      lower.includes("password") ||
      lower.includes("bearer") ||
      lower.includes("secret") ||
      lower.includes("api_key") ||
      lower.includes("token")
    );
  }
}

export const strategyMemory = new StrategyMemoryStore();
