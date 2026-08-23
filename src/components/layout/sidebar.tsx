"use client";

/**
 * Conversation history & projects navigation rail.
 */

import { useState } from "react";
import Link from "next/link";
import {
  Database,
  FolderKanban,
  FolderPlus,
  MessageSquarePlus,
  Pin,
  PinOff,
  Plus,
  Search,
  Settings,
  Trash2,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { BravienWordmark } from "@/components/bravien/mark";
import { Button } from "@/components/ui/button";
import type { UseConversationsResult } from "@/hooks/use-conversations";
import { useProjects } from "@/hooks/use-projects";
import { cn } from "@/lib/utils";

function relativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return "";
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(then).toLocaleDateString();
}

export interface SidebarProps {
  history: UseConversationsResult;
  activeId: string | null;
  activeProjectId?: string | null;
  onClose?: () => void;
  onNavigate?: () => void;
  className?: string;
}

export function Sidebar({
  history,
  activeId,
  activeProjectId,
  onClose,
  onNavigate,
  className,
}: SidebarProps) {
  const [query, setQuery] = useState("");
  const [isCreatingProject, setIsCreatingProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDesc, setNewProjectDesc] = useState("");
  const [creatingBusy, setCreatingBusy] = useState(false);

  const { projects, create: createProject, remove: removeProject } = useProjects();

  const filtered = query.trim()
    ? history.items.filter((c) =>
        c.title.toLowerCase().includes(query.trim().toLowerCase()),
      )
    : history.items;

  async function handleCreateProject(e: React.FormEvent) {
    e.preventDefault();
    if (!newProjectName.trim()) return;
    setCreatingBusy(true);
    try {
      await createProject({
        name: newProjectName.trim(),
        description: newProjectDesc.trim() || null,
      });
      setNewProjectName("");
      setNewProjectDesc("");
      setIsCreatingProject(false);
      toast.success("Project created");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to create project");
    } finally {
      setCreatingBusy(false);
    }
  }

  return (
    <aside
      className={cn(
        "flex h-full w-72 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground",
        className,
      )}
      aria-label="Sidebar navigation"
    >
      <div className="flex items-center justify-between px-3 py-3">
        <Link
          href="/"
          onClick={onNavigate}
          className="rounded-md px-1 py-0.5 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
        >
          <BravienWordmark className="text-sm" />
        </Link>
        {onClose && (
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Close sidebar"
            onClick={onClose}
            className="lg:hidden"
          >
            <X aria-hidden />
          </Button>
        )}
      </div>

      <div className="px-3 space-y-1.5">
        <Button
          className="w-full justify-start"
          size="lg"
          nativeButton={false}
          render={<Link href="/" onClick={onNavigate} />}
        >
          <MessageSquarePlus aria-hidden /> New conversation
        </Button>
      </div>

      {history.state === "ready" && (
        <div className="relative mt-3 px-3">
          <Search
            className="pointer-events-none absolute top-1/2 left-5.5 size-3.5 -translate-y-1/2 text-muted-foreground"
            aria-hidden
          />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search conversations..."
            aria-label="Search conversations"
            className="h-8 w-full rounded-lg border border-sidebar-border bg-background/60 pr-2 pl-8 text-xs outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/40"
          />
        </div>
      )}

      <div className="mt-4 px-3 flex items-center justify-between text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
        <span className="flex items-center gap-1.5">
          <FolderKanban className="size-3.5 text-brand" />
          Projects
        </span>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label="New Project"
          onClick={() => setIsCreatingProject(!isCreatingProject)}
        >
          <Plus className="size-3.5" />
        </Button>
      </div>

      {isCreatingProject && (
        <form onSubmit={handleCreateProject} className="mx-3 mt-1.5 p-2.5 rounded-lg border border-border bg-background/80 space-y-2">
          <input
            autoFocus
            type="text"
            required
            placeholder="Project name"
            value={newProjectName}
            onChange={(e) => setNewProjectName(e.target.value)}
            className="w-full h-7 px-2 text-xs rounded border border-border bg-background outline-none"
          />
          <input
            type="text"
            placeholder="Short description (optional)"
            value={newProjectDesc}
            onChange={(e) => setNewProjectDesc(e.target.value)}
            className="w-full h-7 px-2 text-xs rounded border border-border bg-background outline-none"
          />
          <div className="flex justify-end gap-1.5">
            <Button
              type="button"
              variant="ghost"
              size="xs"
              onClick={() => setIsCreatingProject(false)}
            >
              Cancel
            </Button>
            <Button type="submit" size="xs" disabled={creatingBusy}>
              {creatingBusy ? "Creating..." : "Create"}
            </Button>
          </div>
        </form>
      )}

      {projects.length > 0 && (
        <ul className="mt-1 px-2 space-y-0.5 max-h-36 overflow-y-auto">
          {projects.map((proj) => {
            const active = proj.id === activeProjectId;
            return (
              <li key={proj.id} className="group/proj relative">
                <Link
                  href={`/project/${proj.id}`}
                  onClick={onNavigate}
                  className={cn(
                    "flex items-center justify-between rounded-lg py-1.5 px-2.5 text-xs transition-colors",
                    active
                      ? "bg-brand/10 text-brand font-medium border border-brand/20"
                      : "hover:bg-sidebar-accent/60 text-sidebar-foreground",
                  )}
                >
                  <span className="truncate">{proj.name}</span>
                  <span className="text-[10px] text-muted-foreground ml-1">
                    {proj.conversationCount ?? 0} chats
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      )}

      <div className="mt-4 px-3 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
        Conversations
      </div>

      <nav className="mt-1 min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {history.state === "loading" && (
          <ul className="space-y-1.5 px-1" aria-label="Loading history">
            {[0, 1, 2, 3].map((i) => (
              <li
                key={i}
                className="h-9 animate-pulse rounded-lg bg-sidebar-accent/50"
              />
            ))}
          </ul>
        )}

        {history.state === "unavailable" && (
          <div className="mx-1 rounded-xl border border-sidebar-border bg-background/50 p-3">
            <div className="flex items-center gap-2">
              <Database className="size-3.5 text-muted-foreground" aria-hidden />
              <p className="text-xs font-medium">History is off</p>
            </div>
            <p className="mt-1.5 text-[11px] leading-5 text-muted-foreground">
              {history.reason ??
                "Saving conversations needs a database and a signed-in user."}
            </p>
            <p className="mt-2 text-[11px] leading-5 text-muted-foreground">
              Chatting still works — turns just are not written down.
            </p>
          </div>
        )}

        {history.state === "error" && (
          <div className="mx-1 rounded-xl border border-destructive/30 bg-destructive/5 p-3">
            <p className="text-xs font-medium text-destructive">
              Could not load history
            </p>
            <p className="mt-1 text-[11px] leading-5 text-muted-foreground">
              {history.reason}
            </p>
            <Button
              variant="outline"
              size="xs"
              className="mt-2"
              onClick={() => void history.refresh()}
            >
              Try again
            </Button>
          </div>
        )}

        {history.state === "ready" && filtered.length === 0 && (
          <p className="px-3 py-6 text-center text-xs text-muted-foreground">
            {query.trim()
              ? "Nothing matches that."
              : "No conversations yet. Ask Bravien something."}
          </p>
        )}

        <ul className="space-y-0.5">
          {filtered.map((conversation) => {
            const active = conversation.id === activeId;
            return (
              <li key={conversation.id} className="group/item relative">
                <Link
                  href={`/chat/${conversation.id}`}
                  onClick={onNavigate}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "block rounded-lg py-2 pr-14 pl-2.5 transition-colors",
                    "focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
                    active
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "hover:bg-sidebar-accent/60",
                  )}
                >
                  <span className="flex items-center gap-1.5">
                    {conversation.pinned && (
                      <Pin className="size-3 shrink-0 text-brand" aria-hidden />
                    )}
                    <span className="truncate text-[13px] font-medium">
                      {conversation.title}
                    </span>
                  </span>
                  <span className="mt-0.5 block truncate text-[10px] text-muted-foreground">
                    {relativeTime(conversation.updatedAt)} · {conversation.model}
                  </span>
                </Link>

                <div className="absolute top-1.5 right-1 flex gap-0.5 opacity-0 transition-opacity group-hover/item:opacity-100 focus-within:opacity-100">
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    aria-label={conversation.pinned ? "Unpin" : "Pin"}
                    onClick={() =>
                      void history.togglePin(conversation.id).catch((error) =>
                        toast.error("Could not update", {
                          description:
                            error instanceof Error ? error.message : undefined,
                        }),
                      )
                    }
                  >
                    {conversation.pinned ? (
                      <PinOff aria-hidden />
                    ) : (
                      <Pin aria-hidden />
                    )}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    aria-label={`Delete ${conversation.title}`}
                    onClick={() => {
                      if (
                        !window.confirm(
                          `Delete "${conversation.title}" and its messages? This cannot be undone.`,
                        )
                      ) {
                        return;
                      }
                      void history
                        .remove(conversation.id)
                        .catch((error) =>
                          toast.error("Could not delete", {
                            description:
                              error instanceof Error ? error.message : undefined,
                          }),
                        );
                    }}
                  >
                    <Trash2 aria-hidden />
                  </Button>
                </div>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="border-t border-sidebar-border p-2 space-y-1">
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start"
          nativeButton={false}
          render={<Link href="/tasks" onClick={onNavigate} />}
        >
          <FolderKanban aria-hidden className="mr-2 h-4 w-4" /> Tasks &amp; Workspaces
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start"
          nativeButton={false}
          render={<Link href="/settings" onClick={onNavigate} />}
        >
          <Settings aria-hidden className="mr-2 h-4 w-4" /> Settings
        </Button>
      </div>
    </aside>
  );
}
