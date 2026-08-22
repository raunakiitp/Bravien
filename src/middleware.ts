import NextAuth from "next-auth";
import { authConfig } from "@/auth.config";

/**
 * Edge-safe Auth.js wrapper (providers live in auth.ts).
 * Next.js 16 prefers `proxy.ts`; keep this file until you migrate with
 * `npx @next/codemod@canary middleware-to-proxy .`
 */
const { auth } = NextAuth(authConfig);

export default auth;

export const config = {
  matcher: [
    "/chat/:path*",
    "/settings/:path*",
    "/api/conversations/:path*",
    "/api/chat/:path*",
    "/api/files/:path*",
    "/api/memories/:path*",
  ],
};
