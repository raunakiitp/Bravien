"use client";

import { use, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  FileCode,
  FileText,
  FolderKanban,
  Loader2,
  MessageSquare,
  MessageSquarePlus,
  Plus,
  Save,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { toast } from "sonner";

import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { useConversations } from "@/hooks/use-conversations";
import { useRuntime } from "@/hooks/use-runtime";
import { apiFetch } from "@/types/api";

interface ProjectDetails {
  id: string;
  name: string;
  description: string | null;
  instructions: string | null;
  createdAt: string;
  updatedAt: string;
  conversations: Array<{
    id: string;
    title: string;
    model: string;
    updatedAt: string;
  }>;
  attachments: Array<{
    id: string;
    filename: string;
    mimeType: string;
    size: number;
    createdAt: string;
  }>;
  memories: Array<{
    id: string;
    type: string;
    content: string;
    updatedAt: string;
  }>;
}

export default function ProjectPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const router = useRouter();
  const history = useConversations();
  const { snapshot, loading: runtimeLoading } = useRuntime();

  const [project, setProject] = useState<ProjectDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [instructions, setInstructions] = useState("");
  const [savingInstructions, setSavingInstructions] = useState(false);
  const [uploading, setUploading] = useState(false);

  const fetchProject = useCallback(async () => {
    try {
      const data = await apiFetch<{ project: ProjectDetails }>(`/api/projects/${id}`);
      setProject(data.project);
      setInstructions(data.project.instructions || "");
      setLoading(false);
    } catch (err) {
      setLoading(false);
      toast.error(err instanceof Error ? err.message : "Failed to load project");
    }
  }, [id]);

  useEffect(() => {
    void fetchProject();
  }, [fetchProject]);

  async function handleSaveInstructions() {
    setSavingInstructions(true);
    try {
      await apiFetch(`/api/projects/${id}`, {
        method: "PATCH",
        body: JSON.stringify({ instructions }),
      });
      toast.success("Project instructions saved");
      void fetchProject();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save instructions");
    } finally {
      setSavingInstructions(false);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`/api/projects/${id}/files`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error?.message || data.message || "Upload failed");
      }
      toast.success(`Uploaded ${file.name} (${data.chunkCount} chunks extracted)`);
      void fetchProject();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function handleDeleteFile(fileId: string, filename: string) {
    if (!window.confirm(`Delete ${filename}?`)) return;
    try {
      await apiFetch(`/api/projects/${id}/files/${fileId}`, {
        method: "DELETE",
      });
      toast.success("File deleted");
      void fetchProject();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete file");
    }
  }

  async function handleDeleteProject() {
    if (!window.confirm(`Delete project "${project?.name}" and all its files?`)) return;
    try {
      await apiFetch(`/api/projects/${id}`, { method: "DELETE" });
      toast.success("Project deleted");
      router.push("/");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete project");
    }
  }

  return (
    <AppShell
      history={history}
      snapshot={snapshot}
      runtimeLoading={runtimeLoading}
      activeId={null}
      activeProjectId={id}
    >
      <div className="flex flex-1 flex-col overflow-y-auto bg-background p-6 md:p-10">
        <div className="mx-auto w-full max-w-4xl space-y-8">
          <div className="flex items-center justify-between">
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <ArrowLeft className="size-3.5" /> Back to chat
            </Link>
            <Button
              variant="ghost"
              size="xs"
              className="text-destructive hover:bg-destructive/10"
              onClick={handleDeleteProject}
            >
              <Trash2 className="size-3.5 mr-1" /> Delete Project
            </Button>
          </div>

          {loading ? (
            <div className="space-y-4">
              <div className="h-10 w-60 animate-pulse rounded-lg bg-muted/60" />
              <div className="h-24 w-full animate-pulse rounded-xl bg-muted/40" />
            </div>
          ) : project ? (
            <>
              <div>
                <div className="flex items-center gap-2.5">
                  <FolderKanban className="size-7 text-brand" />
                  <h1 className="font-display text-2xl font-bold tracking-tight">
                    {project.name}
                  </h1>
                </div>
                {project.description && (
                  <p className="mt-1.5 text-sm text-muted-foreground">
                    {project.description}
                  </p>
                )}
              </div>

              {/* Action Bar */}
              <div className="flex items-center gap-3">
                <Button
                  onClick={() => router.push(`/?project=${id}`)}
                  className="gap-2"
                >
                  <MessageSquarePlus className="size-4" /> Start Project Chat
                </Button>

                <label className="cursor-pointer">
                  <input
                    type="file"
                    className="hidden"
                    accept=".txt,.md,.csv,.json,.pdf,.docx"
                    onChange={handleFileUpload}
                    disabled={uploading}
                  />
                  <span className="inline-flex h-9 items-center justify-center gap-2 rounded-lg border border-border bg-background px-3 text-xs font-medium hover:bg-accent transition-colors">
                    {uploading ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <UploadCloud className="size-3.5 text-brand" />
                    )}
                    Upload Document
                  </span>
                </label>
              </div>

              {/* Project Instructions Section */}
              <div className="rounded-2xl border border-border bg-card p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-semibold">Custom Project Instructions</h2>
                    <p className="text-xs text-muted-foreground">
                      Bravien includes these instructions in context for every conversation in this project.
                    </p>
                  </div>
                  <Button
                    size="xs"
                    onClick={handleSaveInstructions}
                    disabled={savingInstructions}
                    className="gap-1.5"
                  >
                    {savingInstructions ? (
                      <Loader2 className="size-3 animate-spin" />
                    ) : (
                      <Save className="size-3" />
                    )}
                    Save Instructions
                  </Button>
                </div>
                <textarea
                  rows={4}
                  value={instructions}
                  onChange={(e) => setInstructions(e.target.value)}
                  placeholder="e.g., You are an expert ML research tutor. Explain concepts clearly and concisely with code examples."
                  className="w-full rounded-lg border border-border bg-background/50 p-3 text-xs outline-none focus-visible:border-ring focus-visible:ring-2"
                />
              </div>

              {/* Documents / Files Section */}
              <div className="rounded-2xl border border-border bg-card p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-semibold">Reference Documents ({project.attachments.length})</h2>
                    <p className="text-xs text-muted-foreground">
                      Uploaded files are parsed, chunked, and automatically retrieved when asking questions.
                    </p>
                  </div>
                </div>

                {project.attachments.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-border py-8 text-center">
                    <FileText className="mx-auto size-8 text-muted-foreground/50" />
                    <p className="mt-2 text-xs font-medium text-muted-foreground">
                      No documents yet
                    </p>
                    <p className="text-[11px] text-muted-foreground/80 mt-0.5">
                      Upload PDF, TXT, Markdown, CSV, JSON to ground conversations.
                    </p>
                  </div>
                ) : (
                  <ul className="divide-y divide-border rounded-xl border border-border">
                    {project.attachments.map((file) => (
                      <li
                        key={file.id}
                        className="flex items-center justify-between p-3 text-xs"
                      >
                        <div className="flex items-center gap-2.5 min-w-0">
                          <FileCode className="size-4 shrink-0 text-brand" />
                          <div className="min-w-0">
                            <p className="font-medium truncate">{file.filename}</p>
                            <p className="text-[10px] text-muted-foreground">
                              {Math.round(file.size / 1024)} KB · {new Date(file.createdAt).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          onClick={() => handleDeleteFile(file.id, file.filename)}
                          className="text-muted-foreground hover:text-destructive"
                        >
                          <Trash2 className="size-3.5" />
                        </Button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              {/* Project Conversations */}
              <div className="rounded-2xl border border-border bg-card p-5 space-y-3">
                <h2 className="text-sm font-semibold">
                  Project Conversations ({project.conversations.length})
                </h2>
                {project.conversations.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No conversations in this project yet.
                  </p>
                ) : (
                  <ul className="divide-y divide-border rounded-xl border border-border">
                    {project.conversations.map((conv) => (
                      <li key={conv.id}>
                        <Link
                          href={`/chat/${conv.id}`}
                          className="flex items-center justify-between p-3 text-xs hover:bg-accent/50 transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            <MessageSquare className="size-3.5 text-brand" />
                            <span className="font-medium">{conv.title}</span>
                          </div>
                          <span className="text-[10px] text-muted-foreground">
                            {new Date(conv.updatedAt).toLocaleDateString()}
                          </span>
                        </Link>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          ) : (
            <p className="text-sm text-destructive">Project not found.</p>
          )}
        </div>
      </div>
    </AppShell>
  );
}
