/**
 * Whether persistence is available.
 *
 * Bravien's core promise is that a local checkpoint answers locally. Requiring a
 * Postgres instance before the user can talk to their own model would break that,
 * so the database is optional: with `DATABASE_URL` set, conversations persist;
 * without it, chat still works and nothing is saved.
 *
 * The UI is told which mode it is in rather than silently losing history.
 */

let probe: Promise<boolean> | null = null;

/** Cheap synchronous check: is a database even configured? */
export function isDatabaseConfigured(): boolean {
  return Boolean(process.env.DATABASE_URL?.trim());
}

/**
 * Can we actually reach it? Configured-but-down is the common local case (the
 * compose stack not started), and it must degrade to ephemeral chat rather than
 * a 500 on every request.
 */
export async function isDatabaseReachable(): Promise<boolean> {
  if (!isDatabaseConfigured()) return false;
  if (probe) return probe;

  probe = (async () => {
    try {
      const { prisma } = await import("@/lib/db/prisma");
      await prisma.$queryRaw`SELECT 1`;
      return true;
    } catch {
      return false;
    }
  })();

  const result = await probe;
  // Retry a failed probe on the next request — the user may have just started
  // the database. A success is stable enough to keep.
  if (!result) probe = null;
  return result;
}

/** Forget the cached probe (tests, or after starting the database). */
export function resetDatabaseProbe(): void {
  probe = null;
}
