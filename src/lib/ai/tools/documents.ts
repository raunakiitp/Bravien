import { z } from "zod";
import { retrieveRelevantContext } from "@/lib/rag/retrieval";
import { prisma } from "@/lib/db/prisma";
import type { ToolDefinition } from "./base";

const searchDocumentsSchema = z.object({
  query: z.string().min(1, "Search query is required"),
  limit: z.number().int().min(1).max(10).optional().default(4),
});

export const searchDocumentsTool: ToolDefinition<typeof searchDocumentsSchema> = {
  name: "search_documents",
  description:
    "Search through project documents, PDFs, notes, and uploaded files for relevant text sections and evidence.",
  category: "DOCUMENT",
  riskLevel: "LOW",
  requiresNetwork: false,
  requiresProjectScope: true,
  mutatesData: false,
  executionTimeoutMs: 5000,
  schema: searchDocumentsSchema,
  execute: async ({ query, limit }, context) => {
    if (!context.projectId) {
      return {
        success: true,
        data: [],
        formattedOutput:
          "No project is active for document search. Attach files to a project or start a project chat to search reference documents.",
      };
    }

    try {
      const hits = await retrieveRelevantContext({
        userId: context.userId,
        projectId: context.projectId,
        query,
        limit,
      });

      if (hits.length === 0) {
        return {
          success: true,
          data: [],
          formattedOutput: `No document excerpts found in project matching "${query}".`,
        };
      }

      const formatted = hits
        .map(
          (h, i) =>
            `[Excerpt ${i + 1}] Source: ${h.filename} (Section ${h.chunkIndex + 1}):\n${h.content}`,
        )
        .join("\n\n");

      return {
        success: true,
        data: hits,
        formattedOutput: `Found ${hits.length} relevant document sections:\n\n${formatted}`,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
        formattedOutput: `Failed to search documents: ${err instanceof Error ? err.message : "Error"}`,
      };
    }
  },
};

const getDocumentContentSchema = z.object({
  filename: z.string().min(1, "Filename is required"),
});

export const getDocumentContentTool: ToolDefinition<typeof getDocumentContentSchema> = {
  name: "get_document_content",
  description: "Retrieve extracted text and metadata of a specific file in the active project.",
  category: "DOCUMENT",
  riskLevel: "LOW",
  requiresNetwork: false,
  requiresProjectScope: true,
  mutatesData: false,
  executionTimeoutMs: 4000,
  schema: getDocumentContentSchema,
  execute: async ({ filename }, context) => {
    if (!context.projectId) {
      return {
        success: false,
        error: "No active project",
        formattedOutput: "No active project to retrieve documents from.",
      };
    }

    try {
      const file = await prisma.attachment.findFirst({
        where: {
          userId: context.userId,
          projectId: context.projectId,
          filename: { equals: filename, mode: "insensitive" },
        },
        select: {
          filename: true,
          mimeType: true,
          size: true,
          extractedText: true,
          status: true,
        },
      });

      if (!file) {
        return {
          success: false,
          error: `File "${filename}" not found in project.`,
          formattedOutput: `Document "${filename}" was not found in the current project.`,
        };
      }

      const textSnippet = file.extractedText
        ? file.extractedText.slice(0, 4000)
        : "(No extracted text available)";

      return {
        success: true,
        data: file,
        formattedOutput: `Document: ${file.filename} (${file.mimeType}, ${Math.round(file.size / 1024)} KB)\n\nContent:\n${textSnippet}`,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
        formattedOutput: `Failed to retrieve document: ${err instanceof Error ? err.message : "Error"}`,
      };
    }
  },
};
