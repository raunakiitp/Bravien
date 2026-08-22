"use client";

/**
 * One turn on screen.
 *
 * The three states are visually distinct on purpose: text the model produced,
 * text the user wrote, and a failure. A failed turn is never dressed as prose —
 * it gets an error panel with the runtime's own code and message, so "the model
 * said nothing" can never be mistaken for "the model said this" (§52, §72).
 */

import { useState } from "react";
import {
  AlertTriangle,
  Check,
  Copy,
  Pencil,
  RefreshCw,
  Square,
  X,
} from "lucide-react";

import { BravienMark } from "@/components/bravien/mark";
import { Markdown } from "@/components/chat/markdown";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { UiMessage } from "@/hooks/use-chat";
import { cn } from "@/lib/utils";

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="ghost"
      size="icon-xs"
      aria-label={copied ? "Copied" : "Copy message"}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1600);
        } catch {
          // Clipboard refused; the text remains selectable.
        }
      }}
    >
      {copied ? (
        <Check className="text-brand" aria-hidden />
      ) : (
        <Copy aria-hidden />
      )}
    </Button>
  );
}

function UserTurn({
  message,
  onEdit,
  editable,
}: {
  message: UiMessage;
  onEdit: (text: string) => void;
  editable: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);

  if (editing) {
    return (
      <div className="flex flex-col items-end gap-2">
        <Textarea
          value={draft}
          autoFocus
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Escape") setEditing(false);
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
              event.preventDefault();
              setEditing(false);
              onEdit(draft);
            }
          }}
          className="max-w-[46rem] bg-card"
          rows={3}
          aria-label="Edit your message"
        />
        <div className="flex gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setDraft(message.content);
              setEditing(false);
            }}
          >
            <X aria-hidden /> Cancel
          </Button>
          <Button
            size="sm"
            disabled={!draft.trim() || draft === message.content}
            onClick={() => {
              setEditing(false);
              onEdit(draft);
            }}
          >
            Send again
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="group/turn flex flex-col items-end gap-1">
      <div className="max-w-[46rem] rounded-2xl rounded-br-md border border-border/70 bg-secondary px-4 py-2.5 text-[15px] leading-7 whitespace-pre-wrap break-words text-secondary-foreground">
        {message.content}
      </div>

      {message.attachments?.length ? (
        <ul className="flex flex-wrap justify-end gap-1.5">
          {message.attachments.map((file) => (
            <li
              key={file.name}
              className="rounded-md border border-border bg-muted/50 px-2 py-0.5 font-mono text-[11px] text-muted-foreground"
            >
              {file.name} · {file.characters.toLocaleString()} chars
            </li>
          ))}
        </ul>
      ) : null}

      <div className="flex gap-0.5 opacity-0 transition-opacity group-hover/turn:opacity-100 focus-within:opacity-100">
        <CopyButton text={message.content} />
        {editable && (
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Edit and resend"
            onClick={() => setEditing(true)}
          >
            <Pencil aria-hidden />
          </Button>
        )}
      </div>
    </div>
  );
}

function TurnError({
  error,
  onRetry,
}: {
  error: { code: string; message: string };
  onRetry?: () => void;
}) {
  return (
    <div
      role="alert"
      className="rounded-xl border border-destructive/35 bg-destructive/5 px-4 py-3"
    >
      <div className="flex items-start gap-2.5">
        <AlertTriangle
          className="mt-0.5 size-4 shrink-0 text-destructive"
          aria-hidden
        />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-destructive">
            Bravien could not answer
          </p>
          <p className="mt-1 text-sm leading-6 text-foreground/80">
            {error.message}
          </p>
          <p className="mt-1.5 font-mono text-[11px] text-muted-foreground">
            {error.code}
          </p>
        </div>
        {onRetry && (
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw aria-hidden /> Retry
          </Button>
        )}
      </div>
    </div>
  );
}

function AssistantTurn({
  message,
  onRegenerate,
  onStop,
  isLast,
  busy,
}: {
  message: UiMessage;
  onRegenerate?: () => void;
  onStop?: () => void;
  isLast: boolean;
  busy: boolean;
}) {
  const streaming = message.status === "streaming" || message.status === "pending";
  const empty = message.content.length === 0;

  return (
    <div className="group/turn border-l-2 border-brand/25 pl-4 sm:pl-5">
      <div className="mb-1.5 flex items-center gap-2">
        <BravienMark className="size-4 text-brand" />
        <span className="font-display text-[13px] font-semibold tracking-tight">
          Bravien
        </span>
        {message.model && (
          <span className="rounded border border-border/70 px-1.5 py-px font-mono text-[10px] text-muted-foreground">
            {message.model}
          </span>
        )}
        {message.status === "stopped" && (
          <span className="text-[11px] text-muted-foreground">
            stopped early
          </span>
        )}
      </div>

      {message.error && empty ? (
        <TurnError error={message.error} onRetry={onRegenerate} />
      ) : (
        <>
          {streaming && empty ? (
            <p
              className="streaming-cursor text-sm text-muted-foreground"
              aria-live="polite"
            >
              Generating
            </p>
          ) : (
            <Markdown
              content={message.content}
              className={cn(streaming && "streaming-cursor")}
            />
          )}

          {/* A failure after partial output: keep the text, flag the break. */}
          {message.error && !empty && (
            <div className="mt-3">
              <TurnError error={message.error} onRetry={onRegenerate} />
            </div>
          )}
        </>
      )}

      <div
        className={cn(
          "mt-2 flex items-center gap-0.5 transition-opacity",
          streaming
            ? "opacity-100"
            : "opacity-0 group-hover/turn:opacity-100 focus-within:opacity-100",
        )}
      >
        {streaming ? (
          onStop && (
            <Button variant="ghost" size="sm" onClick={onStop}>
              <Square aria-hidden /> Stop
            </Button>
          )
        ) : (
          <>
            {!empty && <CopyButton text={message.content} />}
            {isLast && onRegenerate && (
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Regenerate this answer"
                disabled={busy}
                onClick={onRegenerate}
              >
                <RefreshCw aria-hidden />
              </Button>
            )}
            {message.usage?.outputTokens != null && (
              <span className="ml-1 font-mono text-[10px] text-muted-foreground">
                {message.usage.outputTokens} tok
                {message.finishReason && message.finishReason !== "stop"
                  ? ` · ${message.finishReason}`
                  : ""}
              </span>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export function MessageTurn({
  message,
  isLast,
  busy,
  onRegenerate,
  onStop,
  onEdit,
}: {
  message: UiMessage;
  isLast: boolean;
  busy: boolean;
  onRegenerate?: () => void;
  onStop?: () => void;
  onEdit?: (text: string) => void;
}) {
  if (message.role === "user") {
    return (
      <UserTurn
        message={message}
        editable={Boolean(onEdit) && !busy}
        onEdit={(text) => onEdit?.(text)}
      />
    );
  }
  return (
    <AssistantTurn
      message={message}
      isLast={isLast}
      busy={busy}
      onRegenerate={onRegenerate}
      onStop={onStop}
    />
  );
}
