import { NextResponse } from "next/server";
import { z } from "zod";

import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { createProject, listProjects } from "@/lib/db/projects";
import { logger } from "@/lib/observability/logger";
import { badRequest, jsonError, unauthorized } from "@/lib/security/errors";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const createProjectSchema = z
  .object({
    name: z.string().min(1, "Project name is required.").max(200),
    description: z.string().max(1000).optional().nullable(),
    instructions: z.string().max(20000).optional().nullable(),
  })
  .strict();

export async function GET() {
  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(
      503,
      "PERSISTENCE_UNAVAILABLE",
      "Projects require a connected database.",
    );
  }

  try {
    const projects = await listProjects(userId);
    return NextResponse.json({ items: projects });
  } catch (error) {
    logger.error("projects.list_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "PROJECTS_ERROR", "Could not load projects.");
  }
}

export async function POST(request: Request) {
  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(
      503,
      "PERSISTENCE_UNAVAILABLE",
      "Creating projects requires a connected database.",
    );
  }

  let json: unknown;
  try {
    json = await request.json();
  } catch {
    return badRequest("Invalid JSON payload.");
  }

  const parsed = createProjectSchema.safeParse(json);
  if (!parsed.success) {
    return badRequest("Invalid project input.", parsed.error.issues);
  }

  try {
    const project = await createProject({
      userId,
      name: parsed.data.name,
      description: parsed.data.description,
      instructions: parsed.data.instructions,
    });
    return NextResponse.json(project, { status: 201 });
  } catch (error) {
    logger.error("projects.create_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "PROJECT_CREATE_ERROR", "Could not create project.");
  }
}
