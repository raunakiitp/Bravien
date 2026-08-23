import { NextResponse } from "next/server";
import { z } from "zod";

import { hashPassword } from "@/lib/auth/password";
import { isDatabaseReachable } from "@/lib/db/available";
import { prisma } from "@/lib/db/prisma";
import { logger } from "@/lib/observability/logger";
import { badRequest, jsonError } from "@/lib/security/errors";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const registerSchema = z
  .object({
    email: z.string().email("Please provide a valid email address.").max(255),
    password: z
      .string()
      .min(8, "Password must be at least 8 characters long.")
      .max(128, "Password is too long."),
    name: z.string().max(100).optional(),
  })
  .strict();

export async function POST(request: Request) {
  try {
    const reachable = await isDatabaseReachable();
    if (!reachable) {
      return jsonError(
        503,
        "DATABASE_UNAVAILABLE",
        "The database is currently unreachable. Please verify that PostgreSQL is running.",
      );
    }

    let json: unknown;
    try {
      json = await request.json();
    } catch {
      return badRequest("Invalid JSON body.");
    }

    const parsed = registerSchema.safeParse(json);
    if (!parsed.success) {
      const firstIssue = parsed.error.issues[0];
      return badRequest(firstIssue?.message ?? "Invalid registration payload.");
    }

    const { email, password, name } = parsed.data;
    const normalizedEmail = email.toLowerCase().trim();

    const existing = await prisma.user.findUnique({
      where: { email: normalizedEmail },
      select: { id: true },
    });

    if (existing) {
      return jsonError(
        409,
        "USER_ALREADY_EXISTS",
        "An account with this email address already exists. Please sign in instead.",
      );
    }

    const passwordHash = await hashPassword(password);

    const user = await prisma.user.create({
      data: {
        email: normalizedEmail,
        passwordHash,
        name: name?.trim() || normalizedEmail.split("@")[0],
        role: "USER",
      },
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        createdAt: true,
      },
    });

    logger.info("auth.register_success", { userId: user.id, email: user.email });

    return NextResponse.json(
      {
        message: "Account created successfully.",
        user,
      },
      { status: 201 },
    );
  } catch (error) {
    logger.error("auth.register_error", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(
      500,
      "REGISTRATION_FAILED",
      "An unexpected error occurred during account creation.",
    );
  }
}
