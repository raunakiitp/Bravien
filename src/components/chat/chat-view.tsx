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
import {
  ArrowDown,
  BookOpen,
  Code2,
  Compass,
  FileText,
  Info,
  Languages,
  Lightbulb,
  Sparkles,
} from "lucide-react";

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

const STICK_THRESHOLD_PX = 96;

const SUGGESTION_CATEGORIES = [
  {
    icon: Lightbulb,
    title: "Explain something",
    prompt: "Explain machine learning simply with a real-world analogy",
    tag: "Concept",
  },
  {
    icon: Code2,
    title: "Write code",
    prompt: "Write a clean Python function to reverse a string with tests",
    tag: "Coding",
  },
  {
    icon: FileText,
    title: "Analyze a document",
    prompt: "Summarize key architectural trade-offs in local AI systems",
    tag: "Analysis",
  },
  {
    icon: Compass,
    title: "Research & Explore",
    prompt: "Explain modern microservice architecture patterns and pitfalls",
    tag: "Architecture",
  },
  {
    icon: BookOpen,
    title: "Help me plan",
    prompt: "Create a 5-step roadmap to migrate from SQLite to PostgreSQL",
    tag: "Planning",
  },
  {
    icon: Languages,
    title: "Ask in Hinglish",
    prompt: "Machine learning kya hota hai simple words me samjhao",
    tag: "Hinglish",
  },
];

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

  // Follow new tokens only while reader is near bottom
  useLayoutEffect(() => {
    if (stuck) scrollToBottom();
  }, [chat.messages, stuck, scrollToBottom]);

  useEffect(() => {
    scrollToBottom();
  }, [conversationId, scrollToBottom]);

  const handleSend = (text: string, attachments: ChatAttachment[] = []) => {
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
    <div className="relative flex min-h-0 flex-1 flex-col bg-background">
      {/* Top Model Badge Bar */}
      <div className="flex items-center justify-between border-b border-border/60 bg-background/90 px-4 py-2 text-xs text-muted-foreground backdrop-blur sm:px-6">
        <div className="flex items-center gap-2">
          <span className="flex size-2 rounded-full bg-emerald-500" />
          <span className="font-medium text-foreground">
            {model?.name ? model.name : "Bravien-v1"}
          </span>
          <span className="rounded-full border border-border/80 bg-muted/40 px-2 py-0.5 font-mono text-[10px]">
            {model?.device ? `${model.device} · ${model.precision}` : "Local GPU"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="hidden text-[11px] sm:inline">100% Private Offline Intelligence</span>
        </div>
      </div>

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
      >
        <div className="mx-auto w-full max-w-3xl px-4 pt-6 pb-6 sm:px-6">
          {empty ? (
            <div className="flex min-h-[58vh] flex-col justify-center py-6">
              <div className="flex items-center gap-3">
                <div className="flex size-10 items-center justify-center rounded-xl bg-brand/10 text-brand border border-brand/20 shadow-xs">
                  <BravienMark className="size-6" />
                </div>
                <div>
                  <h1 className="font-display text-2xl font-semibold tracking-tight sm:text-3xl">
                    Welcome to Bravien
                  </h1>
                  <p className="text-xs text-muted-foreground">
                    Private intelligence running entirely on your local hardware.
                  </p>
                </div>
              </div>

              {ready && model ? (
                <>
                  <p className="mt-4 max-w-2xl text-[14px] leading-6 text-muted-foreground">
                    Bravien-v1 ({formatParameters(model.parameters)} parameters) is active in {model.precision} on {model.device}. Weights, inference engine, context optimization, and tools run locally — zero data leaves this machine.
                  </p>

                  <div className="mt-4 flex flex-wrap items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="rounded-lg text-xs"
                      onClick={() => setShowDetails((v) => !v)}
                      aria-expanded={showDetails}
                    >
                      <Info className="size-3.5" aria-hidden />
                      {showDetails ? "Hide model specifications" : "View model specifications"}
                    </Button>
                    {isSyntheticCheckpoint(model) && (
                      <span className="rounded-full border border-chart-4/40 bg-chart-4/10 px-2.5 py-0.5 text-xs text-foreground/80">
                        Synthetic verification run
                      </span>
                    )}
                  </div>

                  {showDetails && <ModelCard model={model} className="mt-4" />}

                  {/* Interactive Categorized Suggestion Cards */}
                  <div className="mt-7">
                    <div className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      <Sparkles className="size-3.5 text-brand" /> Get started with an idea
                    </div>
                    <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
                      {SUGGESTION_CATEGORIES.map((item, i) => {
                        const Icon = item.icon;
                        return (
                          <button
                            key={i}
                            type="button"
                            onClick={() => handleSend(item.prompt, [])}
                            className="group flex flex-col justify-between rounded-xl border border-border/80 bg-card p-3.5 text-left transition-all hover:border-brand/50 hover:bg-muted/30 hover:shadow-xs active:scale-98 cursor-pointer"
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex size-7 items-center justify-center rounded-lg bg-muted text-muted-foreground group-hover:bg-brand/10 group-hover:text-brand transition-colors">
                                <Icon className="size-4" />
                              </div>
                              <span className="rounded-md bg-muted/60 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                                {item.tag}
                              </span>
                            </div>
                            <div className="mt-3">
                              <p className="text-xs font-medium text-foreground group-hover:text-brand transition-colors">
                                {item.title}
                              </p>
                              <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-muted-foreground">
                                {item.prompt}
                              </p>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </>
              ) : (
                <div className="mt-6">
                  {runtimeLoading ? (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span className="size-2 rounded-full bg-brand animate-ping" />
                      Checking for active Bravien inference runtime…
                    </div>
                  ) : (
                    <RuntimeDown snapshot={snapshot} />
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-6 pb-4">
              {chat.messages.map((message, index) => (
                <MessageTurn
                  key={message.id}
                  message={message}
                  isLast={index === lastAssistantIndex}
                  busy={chat.busy}
                  onStop={chat.stop}
                  onSuggestionClick={(suggestion) => handleSend(suggestion, [])}
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

      {/* Floating Jump to Latest Button */}
      {!stuck && !empty && (
        <Button
          variant="outline"
          size="sm"
          className="absolute bottom-32 left-1/2 -translate-x-1/2 rounded-full border-border/80 bg-background/90 px-3 py-1 text-xs shadow-md backdrop-blur hover:bg-background"
          aria-label="Jump to latest message"
          onClick={() => {
            setStuck(true);
            scrollToBottom(true);
          }}
        >
          <ArrowDown className="mr-1 size-3.5" aria-hidden />
          Jump to latest
        </Button>
      )}

      {/* Fixed Composer Area */}
      <div
        className={cn(
          "border-t border-border/80 bg-background/90 backdrop-blur",
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
