"use client";

/**
 * Saved conversations.
 *
 * History needs a database and a signed-in user. When either is missing the API
 * answers `PERSISTENCE_UNAVAILABLE`, and that is surfaced as "history is off"
 * rather than as an empty list — an empty list would read as "you have never
 * talked to Bravien", which is a different and false statement (§72).
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { ConversationDTO } from "@/types";
import {
  ApiError,
  apiFetch,
  type ConversationListResponse,
} from "@/types/api";

export type HistoryState = "loading" | "ready" | "unavailable" | "error";

export interface UseConversationsResult {
  items: ConversationDTO[];
  state: HistoryState;
  /** Why history is unavailable, in the API's own words. */
  reason: string | null;
  refresh: () => Promise<void>;
  remove: (id: string) => Promise<void>;
  rename: (id: string, title: string) => Promise<void>;
  togglePin: (id: string) => Promise<void>;
  /** Insert a conversation the chat loop just created, without a round trip. */
  upsert: (conversation: ConversationDTO) => void;
}

export function useConversations(): UseConversationsResult {
  const [items, setItems] = useState<ConversationDTO[]>([]);
  const [state, setState] = useState<HistoryState>("loading");
  const [reason, setReason] = useState<string | null>(null);
  const mounted = useRef(true);

  /**
   * Load the list.
   *
   * A promise chain rather than `async`/`await` so the state updates happen in
   * callbacks — mounting must not schedule a re-render inside the effect flush.
   */
  const refresh = useCallback(() => {
    return apiFetch<ConversationListResponse>("/api/conversations?take=60")
      .then((data) => {
        if (!mounted.current) return;
        setItems(data.items);
        setState("ready");
        setReason(null);
      })
      .catch((cause: unknown) => {
        if (!mounted.current) return;
        if (cause instanceof ApiError) {
          // 401 and 503 both mean "no history here", for different reasons; both
          // are expected in a local install and neither is an error to shout about.
          const expected =
            cause.code === "PERSISTENCE_UNAVAILABLE" || cause.status === 401;
          setState(expected ? "unavailable" : "error");
          setReason(cause.message);
        } else {
          setState("error");
          setReason(cause instanceof Error ? cause.message : String(cause));
        }
      });
  }, []);

  useEffect(() => {
    mounted.current = true;
    void refresh();
    return () => {
      mounted.current = false;
    };
  }, [refresh]);

  const remove = useCallback(async (id: string) => {
    await apiFetch<void>(`/api/conversations/${id}`, { method: "DELETE" });
    setItems((current) => current.filter((c) => c.id !== id));
  }, []);

  const patch = useCallback(
    async (id: string, body: Record<string, unknown>) => {
      const updated = await apiFetch<ConversationDTO>(
        `/api/conversations/${id}`,
        { method: "PATCH", body: JSON.stringify(body) },
      );
      setItems((current) =>
        current
          .map((c) => (c.id === id ? updated : c))
          .sort(sortConversations),
      );
    },
    [],
  );

  const rename = useCallback(
    (id: string, title: string) => patch(id, { title }),
    [patch],
  );

  const togglePin = useCallback(
    async (id: string) => {
      const current = items.find((c) => c.id === id);
      if (!current) return;
      await patch(id, { pinned: !current.pinned });
    },
    [items, patch],
  );

  const upsert = useCallback((conversation: ConversationDTO) => {
    setItems((current) => {
      const without = current.filter((c) => c.id !== conversation.id);
      return [conversation, ...without].sort(sortConversations);
    });
  }, []);

  return { items, state, reason, refresh, remove, rename, togglePin, upsert };
}

/** Pinned first, then most recently touched — the same order the API uses. */
function sortConversations(a: ConversationDTO, b: ConversationDTO): number {
  if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
  return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
}
