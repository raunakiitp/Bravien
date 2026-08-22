import type { ApiError } from "@/types";

export function createApiError(
  code: string,
  message: string,
  details?: unknown,
): ApiError {
  return {
    error: {
      code,
      message,
      ...(details !== undefined ? { details } : {}),
    },
  };
}

export function jsonError(
  status: number,
  code: string,
  message: string,
  details?: unknown,
): Response {
  return Response.json(createApiError(code, message, details), { status });
}

export function unauthorized(message = "Unauthorized"): Response {
  return jsonError(401, "UNAUTHORIZED", message);
}

export function forbidden(message = "Forbidden"): Response {
  return jsonError(403, "FORBIDDEN", message);
}

export function notFound(message = "Not found"): Response {
  return jsonError(404, "NOT_FOUND", message);
}

export function badRequest(message: string, details?: unknown): Response {
  return jsonError(400, "BAD_REQUEST", message, details);
}

export function tooManyRequests(message = "Rate limit exceeded"): Response {
  return jsonError(429, "RATE_LIMITED", message);
}

export function internalError(message = "Internal server error"): Response {
  return jsonError(500, "INTERNAL_ERROR", message);
}
