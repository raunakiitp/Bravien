import { auth } from "@/auth";
import type { Session } from "next-auth";

export class UnauthorizedError extends Error {
  constructor(message = "Unauthorized") {
    super(message);
    this.name = "UnauthorizedError";
  }
}

export async function getCurrentUser(): Promise<Session["user"] | null> {
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
