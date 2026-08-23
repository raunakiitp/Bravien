"use client";

/**
 * The chat surface.
 *
 * Owns scroll behaviour and the empty state; the turn loop lives in `useChat` and
 * the model facts in `useRuntime`. When there is no checkpoint the composer is
 * disabled rather than accepting a message that could not be answered — pretending
 * to send would be the start of pretending to reply (§52).
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { ArrowDown, Info } from "lucide-react";

import { BravienMark } from "@/components/bravien/mark";
import { Composer } from "@/components/chat/composer";
import { MessageTurn } from "@/components/chat/message";
import {
  ModelCard,
  RuntimeDown,
  formatParameters,
  isSyntheticCheckpoint,
} from "@/components/runtime/status";
import { Button } from "@/components/ui/button";
import { useChat, type ChatAttachment } from "@/hooks/use-chat";
import { cn } from "@/lib/utils";
import type { MessageDTO } from "@/types";
import type { RuntimeSnapshot } from "@/types/api";

/** Within this many pixels of the bottom counts as "following the output". */
const STICK_THRESHOLD_PX = 96;

export interface ChatViewProps {
  snapshot: RuntimeSnapshot | null;
  runtimeLoading: boolean;
  conversationId?: string | null;
  projectId?: string | null;
  initialMessages?: MessageDTO[];
  onConversationCreated?: (id: string, title?: string) => void;
}

export function ChatView({
  snapshot,
  runtimeLoading,
  conversationId = null,
  projectId = null,
  initialMessages,
  onConversationCreated,
}: ChatViewProps) {
  const activeModelId = snapshot?.models[0]?.id ?? null;

  const chat = useChat({
    conversationId,
    projectId,
    initialMessages,
    modelId: activeModelId,
    onConversationCreated,
  });

  const scrollRef = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState(true);
  const [showDetails, setShowDetails] = useState(false);

  const ready = Boolean(snapshot?.runtime.modelLoaded && snapshot.models.length);

  const scrollToBottom = useCallback((smooth = false) => {
    const node = scrollRef.current;
    if (!node) return;
    node.scrollTo({
      top: node.scrollHeight,
      behavior: smooth ? "smooth" : "auto",
    });
  }, []);

  const onScroll = useCallback(() => {
    const node = scrollRef.current;
    if (!node) return;
    const distance = node.scrollHeight - node.scrollTop - node.clientHeight;
    setStuck(distance <= STICK_THRESHOLD_PX);
  }, []);

  // Follow new tokens only while the reader is at the bottom; yanking the view
  // back while they are reading earlier output is worse than a stale scroll.
  useLayoutEffect(() => {
    if (stuck) scrollToBottom();
  }, [chat.messages, stuck, scrollToBottom]);

  useEffect(() => {
    scrollToBottom();
  }, [conversationId, scrollToBottom]);

  const handleSend = (text: string, attachments: ChatAttachment[]) => {
    setStuck(true);
    void chat.send(text, { attachments });
  };

  const lastAssistantIndex = (() => {
    for (let i = chat.messages.length - 1; i >= 0; i -= 1) {
      if (chat.messages[i].role === "assistant") return i;
    }
    return -1;
  })();

  const empty = chat.messages.length === 0;
  const model = snapshot?.activeModel ?? null;

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
      >
        <div className="mx-auto w-full max-w-3xl px-4 pt-6 pb-4 sm:px-6">
          {empty ? (
            <div className="flex min-h-[52vh] flex-col justify-center py-8">
              <BravienMark className="size-9 text-brand" />
              <h1 className="mt-4 font-display text-3xl font-semibold tracking-tight sm:text-4xl">
                Bravien
              </h1>

              {ready && model ? (
                <>
                  <p className="mt-3 max-w-xl text-[15px] leading-7 text-muted-foreground">
                    A {formatParameters(model.parameters)}-parameter transformer
                    trained from scratch in this repository, running on{" "}
                    {model.device} at {model.precision}. Its tokenizer, weights and
                    inference engine are its own — nothing in this answer path
                    leaves the machine.
                  </p>

                  <div className="mt-5 flex flex-wrap items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowDetails((v) => !v)}
                      aria-expanded={showDetails}
                    >
                      <Info aria-hidden />
                      {showDetails ? "Hide checkpoint detail" : "What is loaded"}
                    </Button>
                    {isSyntheticCheckpoint(model) && (
                      <span className="rounded-full border border-chart-4/40 bg-chart-4/10 px-2.5 py-1 text-xs text-foreground/80">
                        Trained on synthetic text — a pipeline proof, not an
                        assistant
                      </span>
                    )}
                  </div>

                  {showDetails && <ModelCard model={model} className="mt-4" />}
                </>
              ) : (
                <div className="mt-4">
                  {runtimeLoading ? (
                    <p className="text-sm text-muted-foreground">
                      Checking for a running Bravien runtime…
                    </p>
                  ) : (
                    <RuntimeDown snapshot={snapshot} />
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-7 pb-4">
              {chat.messages.map((message, index) => (
                <MessageTurn
                  key={message.id}
                  message={message}
                  isLast={index === lastAssistantIndex}
                  busy={chat.busy}
                  onStop={chat.stop}
                  onRegenerate={
                    index === lastAssistantIndex && !chat.busy
                      ? () => void chat.regenerate()
                      : undefined
                  }
                  onEdit={
                    message.role === "user" && !chat.busy
                      ? (text) => void chat.editAndResend(message.id, text)
                      : undefined
                  }
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {!stuck && !empty && (
        <Button
          variant="outline"
          size="icon-sm"
          className="absolute bottom-32 left-1/2 -translate-x-1/2 rounded-full shadow-md"
          aria-label="Jump to the latest message"
          onClick={() => {
            setStuck(true);
            scrollToBottom(true);
          }}
        >
          <ArrowDown aria-hidden />
        </Button>
      )}

      <div
        className={cn(
          "border-t border-border bg-background/85 backdrop-blur",
          "px-4 pt-3 pb-4 sm:px-6",
        )}
      >
        <div className="mx-auto w-full max-w-3xl">
          {chat.persisted === false && !empty && (
            <p className="mb-2 px-1 text-[11px] text-muted-foreground">
              This conversation is not being saved — no database is configured.
            </p>
          )}
          <Composer
            onSend={handleSend}
            onStop={chat.stop}
            busy={chat.busy}
            disabled={!ready}
            disabledReason={
              runtimeLoading
                ? "Checking for a running Bravien runtime…"
                : snapshot?.runtime.status === "unreachable"
                  ? "The Bravien runtime is not running, so there is nothing to answer with."
                  : "No checkpoint is loaded, so Bravien has no weights to answer from."
            }
            uploadsEnabled={Boolean(snapshot?.features.file_uploads)}
            autoFocus={ready}
          />
        </div>
      </div>
    </div>
  );
}
