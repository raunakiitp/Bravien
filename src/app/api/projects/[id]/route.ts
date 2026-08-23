import { NextResponse } from "next/server";
import { z } from "zod";

import { requireUser, UnauthorizedError } from "@/lib/auth/session";
import { isDatabaseReachable } from "@/lib/db/available";
import { deleteProject, getProject, updateProject } from "@/lib/db/projects";
import { logger } from "@/lib/observability/logger";
import {
  badRequest,
  jsonError,
  notFound,
  unauthorized,
} from "@/lib/security/errors";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const patchProjectSchema = z
  .object({
    name: z.string().min(1).max(200).optional(),
    description: z.string().max(1000).optional().nullable(),
    instructions: z.string().max(20000).optional().nullable(),
  })
  .strict();

export async function GET(
  _request: Request,
  props: { params: Promise<{ id: string }> },
) {
  const { id } = await props.params;
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
      "Database unavailable.",
    );
  }

  const project = await getProject(userId, id);
  if (!project) {
    return notFound("Project not found.");
  }

  return NextResponse.json({ project });
}

export async function PATCH(
  request: Request,
  props: { params: Promise<{ id: string }> },
) {
  const { id } = await props.params;
  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(503, "PERSISTENCE_UNAVAILABLE", "Database unavailable.");
  }

  let json: unknown;
  try {
    json = await request.json();
  } catch {
    return badRequest("Invalid JSON payload.");
  }

  const parsed = patchProjectSchema.safeParse(json);
  if (!parsed.success) {
    return badRequest("Invalid update payload.", parsed.error.issues);
  }

  try {
    const updated = await updateProject(userId, id, parsed.data);
    if (!updated) {
      return notFound("Project not found.");
    }
    return NextResponse.json(updated);
  } catch (error) {
    logger.error("project.patch_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "PROJECT_UPDATE_ERROR", "Could not update project.");
  }
}

export async function DELETE(
  _request: Request,
  props: { params: Promise<{ id: string }> },
) {
  const { id } = await props.params;
  let userId: string;
  try {
    userId = (await requireUser()).id!;
  } catch (error) {
    if (error instanceof UnauthorizedError) return unauthorized();
    throw error;
  }

  if (!(await isDatabaseReachable())) {
    return jsonError(503, "PERSISTENCE_UNAVAILABLE", "Database unavailable.");
  }

  try {
    const success = await deleteProject(userId, id);
    if (!success) {
      return notFound("Project not found.");
    }
    return new NextResponse(null, { status: 204 });
  } catch (error) {
    logger.error("project.delete_failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return jsonError(500, "PROJECT_DELETE_ERROR", "Could not delete project.");
  }
}
