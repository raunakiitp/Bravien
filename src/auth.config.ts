import type { NextAuthConfig } from "next-auth";

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
      const { pathname } = request.nextUrl;
      const isLoggedIn = !!auth?.user;

      const isProtected =
        pathname.startsWith("/chat") ||
        pathname.startsWith("/settings") ||
        pathname.startsWith("/api/conversations") ||
        pathname.startsWith("/api/chat") ||
        pathname.startsWith("/api/files") ||
        pathname.startsWith("/api/memories");

      if (isProtected) {
        return isLoggedIn;
      }

      return true;
    },
  },
} satisfies NextAuthConfig;
