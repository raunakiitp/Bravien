/**
 * Whether Bravien requires a sign-in.
 *
 * Bravien's default shape is one person, one machine, one checkpoint — the point
 * of the whole architecture is that it answers with no account and no network
 * anywhere in the path (§2). Demanding a login in that setup would lock the owner
 * out of their own weights, so authentication switches on only when it has been
 * configured deliberately: somewhere to keep accounts (`DATABASE_URL`) and a
 * secret to sign sessions with (`AUTH_SECRET`).
 *
 * `BRAVIEN_REQUIRE_AUTH` overrides the inference in either direction. Forcing it
 * on without a database does not create accounts — it just makes every protected
 * route unreachable — so that combination is treated as off.
 *
 * Edge-safe: reads `process.env` and nothing else, because the proxy/middleware
 * runtime imports it.
 */

function truthy(raw: string | undefined): boolean | null {
  const value = raw?.trim().toLowerCase();
  if (value === undefined || value === "") return null;
  if (["1", "true", "yes", "on"].includes(value)) return true;
  if (["0", "false", "no", "off"].includes(value)) return false;
  return null;
}

/**
 * Can this install have accounts at all?
 *
 * Auth.js needs a secret to verify a session cookie and throws `MissingSecret`
 * without one, so this also answers "is it safe to call `auth()`". A database is
 * required too: that is where users live.
 */
export function accountsConfigured(): boolean {
  return (
    Boolean(process.env.DATABASE_URL?.trim()) &&
    Boolean(process.env.AUTH_SECRET?.trim() || process.env.NEXTAUTH_SECRET?.trim())
  );
}

/**
 * Must a visitor sign in before reaching account-only routes?
 *
 * Distinct from {@link accountsConfigured}: an install can have accounts while
 * still leaving everything open, which is what `BRAVIEN_REQUIRE_AUTH=off` means.
 */
export function authRequired(): boolean {
  const override = truthy(process.env.BRAVIEN_REQUIRE_AUTH);
  if (override === false) return false;
  // An explicit `on` still needs somewhere to keep accounts to mean anything.
  return accountsConfigured();
}

/**
 * Route prefixes that cannot work without a user account.
 *
 * `/api/chat` is deliberately absent: a turn runs with or without a session, and
 * the route degrades to un-saved history instead of refusing.
 */
export const ACCOUNT_ROUTES = [
  "/settings",
  "/api/conversations",
  "/api/memories",
] as const;

export function needsAccount(pathname: string): boolean {
  return ACCOUNT_ROUTES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
  );
}
