import { auth } from "@/auth";
import type { Session } from "next-auth";

import { accountsConfigured } from "@/lib/auth/policy";

export class UnauthorizedError extends Error {
  constructor(message = "Unauthorized") {
    super(message);
    this.name = "UnauthorizedError";
  }
}

/**
 * Who is asking, or null.
 *
 * Guarded on configuration: `auth()` throws `MissingSecret` when no secret is
 * set, and in the default single-machine install there are no accounts at all.
 * "Nobody is signed in" is the correct answer there, not a 500 — routes that
 * need an account already degrade gracefully on null.
 */
export async function getCurrentUser(): Promise<Session["user"] | null> {
  if (!accountsConfigured()) return null;
  const session = await auth();
  return session?.user ?? null;
}

export async function requireUser(): Promise<Session["user"]> {
  const user = await getCurrentUser();
  if (!user?.id) {
    throw new UnauthorizedError();
  }
  return user;
}
