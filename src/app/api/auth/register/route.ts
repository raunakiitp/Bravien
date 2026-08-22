import { z } from "zod";
import { hashPassword } from "@/lib/auth/password";
import { isDatabaseReachable } from "@/lib/db/available";
import { prisma } from "@/lib/db/prisma";
import { badRequest, jsonError } from "@/lib/security/errors";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const registerSchema = z.object({
  name: z.string().min(1, "Name is required").max(100).optional(),
  email: z.string().email("Invalid email address"),
  password: z.string().min(6, "Password must be at least 6 characters").max(100),
});

export async function POST(request: Request): Promise<Response> {
  const reachable = await isDatabaseReachable();
  if (!reachable) {
    return jsonError(
      503,
      "PERSISTENCE_UNAVAILABLE",
      "Database is not reachable. Ensure PostgreSQL is running and DATABASE_URL is set in .env.",
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return badRequest("Request body must be valid JSON.");
  }

  const parsed = registerSchema.safeParse(body);
  if (!parsed.success) {
    return badRequest("Invalid registration payload.", parsed.error.issues);
  }

  const { name, email, password } = parsed.data;
  const normalizedEmail = email.trim().toLowerCase();

  const existing = await prisma.user.findUnique({
    where: { email: normalizedEmail },
    select: { id: true },
  });

  if (existing) {
    return badRequest("An account with this email already exists.");
  }

  const passwordHash = await hashPassword(password);

  const user = await prisma.user.create({
    data: {
      email: normalizedEmail,
      name: name?.trim() || normalizedEmail.split("@")[0],
      passwordHash,
      role: "USER",
    },
    select: {
      id: true,
      email: true,
      name: true,
      createdAt: true,
    },
  });

  return Response.json({ user }, { status: 201 });
}
