"use client";

/**
 * The workspace: shell, history rail, and one chat surface.
 *
 * Both `/` and `/chat/[id]` render this. When a brand-new conversation gets saved
 * mid-turn the URL is updated with `history.replaceState` rather than a router
 * navigation — a navigation would remount the chat and throw away the answer
 * currently streaming into it.
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

import { ChatView } from "@/components/chat/chat-view";
import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { useConversations } from "@/hooks/use-conversations";
import { useRuntime } from "@/hooks/use-runtime";
import type { MessageDTO } from "@/types";
import {
  ApiError,
  apiFetch,
  type ConversationDetailResponse,
} from "@/types/api";

type LoadState =
  | { kind: "none" }
  | { kind: "loading" }
  | { kind: "loaded"; data: ConversationDetailResponse }
  | { kind: "error"; message: string; code: string };

export function ChatWorkspace({
  conversationId = null,
}: {
  conversationId?: string | null;
}) {
  const searchParams = useSearchParams();
  const projectParam = searchParams.get("project");

  const runtime = useRuntime();
  const history = useConversations();

  // Tagged with the conversation it belongs to, so a slow response cannot land
  // on a conversation the reader has already navigated away from.
  const [result, setResult] = useState<{ id: string; state: LoadState } | null>(
    null,
  );
  /** Set when a turn creates a conversation; there is no navigation to read it from. */
  const [createdId, setCreatedId] = useState<string | null>(null);

  const activeId = conversationId ?? createdId;

  // Derived rather than stored: no effect needs to reset it on navigation.
  const load: LoadState = !conversationId
    ? { kind: "none" }
    : result?.id === conversationId
      ? result.state
      : { kind: "loading" };

  useEffect(() => {
    if (!conversationId) return;
    let cancelled = false;
    apiFetch<ConversationDetailResponse>(`/api/conversations/${conversationId}`)
      .then((data) => {
        if (!cancelled) {
          setResult({ id: conversationId, state: { kind: "loaded", data } });
        }
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setResult({
          id: conversationId,
          state: {
            kind: "error",
            code: cause instanceof ApiError ? cause.code : "LOAD_FAILED",
            message: cause instanceof Error ? cause.message : String(cause),
          },
        });
      });
    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  const refreshHistory = history.refresh;
  const onConversationCreated = useCallback(
    (id: string) => {
      setCreatedId(id);
      // Shallow URL update: a router navigation here would remount the chat and
      // discard the answer still streaming into it.
      window.history.replaceState(null, "", `/chat/${id}`);
      void refreshHistory();
    },
    [refreshHistory],
  );

  const initialMessages: MessageDTO[] | undefined =
    load.kind === "loaded" ? load.data.messages : undefined;

  const title =
    load.kind === "loaded"
      ? load.data.conversation.title
      : load.kind === "loading"
        ? "Loading…"
        : undefined;

  return (
    <AppShell
      history={history}
      snapshot={runtime.snapshot}
      runtimeLoading={runtime.loading}
      onRefreshRuntime={() => void runtime.refresh()}
      activeId={activeId}
      title={title}
    >
      {load.kind === "error" ? (
        <div className="mx-auto flex w-full max-w-xl flex-1 flex-col justify-center px-6">
          <div className="rounded-2xl border border-border bg-card p-6">
            <h2 className="font-display text-lg font-semibold tracking-tight">
              {load.code === "NOT_FOUND"
                ? "That conversation is not here"
                : "Could not open that conversation"}
            </h2>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              {load.message}
            </p>
            <p className="mt-2 font-mono text-[11px] text-muted-foreground">
              {load.code}
            </p>
            <Button
              className="mt-4"
              nativeButton={false}
              render={<Link href="/" />}
            >
              Start a new conversation
            </Button>
          </div>
        </div>
      ) : load.kind === "loading" ? (
        <div
          className="mx-auto w-full max-w-3xl flex-1 space-y-6 px-4 py-8 sm:px-6"
          aria-label="Loading conversation"
        >
          {[0, 1, 2].map((i) => (
            <div key={i} className="space-y-2">
              <div className="ml-auto h-12 w-2/3 animate-pulse rounded-2xl bg-secondary" />
              <div className="h-24 w-full animate-pulse rounded-xl bg-muted/60" />
            </div>
          ))}
        </div>
      ) : (
        <ChatView
          // Remount when the conversation changes so no turn leaks between them.
          key={conversationId ?? `new-${projectParam ?? "default"}`}
          snapshot={runtime.snapshot}
          runtimeLoading={runtime.loading}
          conversationId={conversationId}
          projectId={projectParam}
          initialMessages={initialMessages}
          onConversationCreated={onConversationCreated}
        />
      )}
    </AppShell>
  );
}
