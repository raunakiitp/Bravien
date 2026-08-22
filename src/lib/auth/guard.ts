/**
 * The entry check for routes that store things per person.
 *
 * Two conditions have to hold before such a route can do anything: a database to
 * write to, and a user to attribute rows to. They fail together in the default
 * install — no `DATABASE_URL` means no accounts either — so the order the checks
 * run in decides what the operator is told.
 *
 * The database is checked first, deliberately. "Not signed in" would be true but
 * useless: it points at a sign-in page belonging to a system with nowhere to keep
 * accounts. "No database is configured, here is the command" is the half of the
 * answer that can be acted on.
 */

import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { jsonError, unauthorized } from "@/lib/security/errors";

/**
 * The caller's id, or the response to return instead.
 *
 * `subject` names the thing being stored, so the failure reads as a sentence
 * about that feature rather than about plumbing. It is surfaced to the user
 * verbatim — the history rail and the memory panel both print it.
 */
export async function accountScope(subject: string): Promise<string | Response> {
  if (!(await isDatabaseReachable())) {
    return jsonError(
      503,
      "PERSISTENCE_UNAVAILABLE",
      `${subject} needs a database. Set DATABASE_URL in .env and run \`npm run db:push\`.`,
    );
  }

  try {
    return (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) {
      // A database exists, so signing in is genuinely the missing step here.
      return unauthorized(`Sign in to use ${subject.toLowerCase()}.`);
    }
    throw error;
  }
}
