import type { NextAuthConfig } from "next-auth";

import { authRequired, needsAccount } from "@/lib/auth/policy";

/**
 * Edge-compatible Auth.js config (no Prisma / Node-only imports).
 * Used by middleware/proxy; full providers live in `auth.ts`.
 */
export const authConfig = {
  pages: {
    signIn: "/login",
  },
  providers: [],
  session: {
    strategy: "jwt",
  },
  callbacks: {
    authorized({ auth, request }) {
      // Without configured accounts there is nobody to be, and redirecting to a
      // sign-in page would strand the owner outside their own runtime.
      if (!authRequired()) return true;
      if (!needsAccount(request.nextUrl.pathname)) return true;
      return Boolean(auth?.user);
    },
  },
} satisfies NextAuthConfig;
