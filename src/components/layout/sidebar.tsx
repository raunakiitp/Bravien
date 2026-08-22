"use client";

/**
 * Conversation history rail.
 *
 * Rendered as a fixed rail on wide screens and a slide-over on narrow ones. When
 * persistence is off the list is replaced by an explanation of what to configure,
 * because the honest answer to "where is my history" is "it was never saved".
 */

import { useState } from "react";
import Link from "next/link";
import {
  Database,
  MessageSquarePlus,
  Pin,
  PinOff,
  Search,
  Settings,
  Trash2,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { BravienWordmark } from "@/components/bravien/mark";
import { Button } from "@/components/ui/button";
import type { UseConversationsResult } from "@/hooks/use-conversations";
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
  onClose?: () => void;
  /** Fired when a link in here is followed, so a slide-over can close itself. */
  onNavigate?: () => void;
  className?: string;
}

export function Sidebar({
  history,
  activeId,
  onClose,
  onNavigate,
  className,
}: SidebarProps) {
  const [query, setQuery] = useState("");

  const filtered = query.trim()
    ? history.items.filter((c) =>
        c.title.toLowerCase().includes(query.trim().toLowerCase()),
      )
    : history.items;

  return (
    <aside
      className={cn(
        "flex h-full w-72 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground",
        className,
      )}
      aria-label="Conversation history"
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
            aria-label="Close history"
            onClick={onClose}
            className="lg:hidden"
          >
            <X aria-hidden />
          </Button>
        )}
      </div>

      <div className="px-3">
        <Button
          className="w-full justify-start"
          size="lg"
          nativeButton={false}
          render={<Link href="/" onClick={onNavigate} />}
        >
          <MessageSquarePlus aria-hidden /> New conversation
        </Button>
      </div>

      {history.state === "ready" && history.items.length > 4 && (
        <div className="relative mt-3 px-3">
          <Search
            className="pointer-events-none absolute top-1/2 left-5.5 size-3.5 -translate-y-1/2 text-muted-foreground"
            aria-hidden
          />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Filter conversations"
            aria-label="Filter conversations"
            className="h-8 w-full rounded-lg border border-sidebar-border bg-background/60 pr-2 pl-8 text-xs outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/40"
          />
        </div>
      )}

      <nav className="mt-3 min-h-0 flex-1 overflow-y-auto px-2 pb-2">
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
                      // Deleting messages is not undoable, so it is confirmed.
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

      <div className="border-t border-sidebar-border p-2">
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start"
          nativeButton={false}
          render={<Link href="/settings" onClick={onNavigate} />}
        >
          <Settings aria-hidden /> Settings
        </Button>
      </div>
    </aside>
  );
}
