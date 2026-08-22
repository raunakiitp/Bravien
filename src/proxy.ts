import NextAuth from "next-auth";
import { authConfig } from "@/auth.config";

/**
 * Edge-safe Auth.js wrapper (providers live in auth.ts).
 *
 * This is the `proxy` file convention that replaced `middleware` in Next 16.
 * Whether a signed-in user is required at all is decided by `authConfig`, which
 * reads the environment: with no database configured there are no accounts, and
 * gating the owner's own runtime behind a login would be absurd (§2).
 */
const { auth } = NextAuth(authConfig);

export default auth;

/**
 * Page routes that need a signed-in user. API routes are deliberately not here:
 * a redirect to an HTML sign-in page is a useless answer to `fetch`, so they
 * return their own 401 JSON from `requireUser()` instead.
 *
 * `/api/chat` is not protected at all — it runs with or without a session and
 * simply does not save history when there is none.
 */
export const config = {
  matcher: ["/settings/:path*"],
};
