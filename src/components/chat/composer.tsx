"use client";

/**
 * The composer.
 *
 * Attachments go through `/api/files`, which extracts text server-side and
 * rejects anything it cannot read. What comes back is what the model will
 * actually see, and it is shown as a chip with a character count — no file is
 * silently accepted and then ignored (§72).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUp, Loader2, Paperclip, Square, X } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import type { ChatAttachment } from "@/hooks/use-chat";
import { cn } from "@/lib/utils";
import { apiErrorFrom, type UploadResponse } from "@/types/api";

const MAX_ROWS_PX = 280;

export interface ComposerProps {
  onSend: (text: string, attachments: ChatAttachment[]) => void;
  onStop: () => void;
  busy: boolean;
  disabled?: boolean;
  disabledReason?: string;
  uploadsEnabled?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
}

export function Composer({
  onSend,
  onStop,
  busy,
  disabled = false,
  disabledReason,
  uploadsEnabled = false,
  placeholder = "Ask Bravien something",
  autoFocus = false,
}: ComposerProps) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [uploading, setUploading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Grow with the content up to a ceiling, then scroll inside.
  const resize = useCallback(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, MAX_ROWS_PX)}px`;
  }, []);

  useEffect(resize, [value, resize]);

  useEffect(() => {
    if (autoFocus) textareaRef.current?.focus();
  }, [autoFocus]);

  const submit = () => {
    const text = value.trim();
    if (!text || busy || disabled) return;
    onSend(text, attachments);
    setValue("");
    setAttachments([]);
    // Return focus so a conversation can be held entirely from the keyboard.
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        const form = new FormData();
        form.append("file", file);
        const response = await fetch("/api/files", {
          method: "POST",
          body: form,
        });
        if (!response.ok) {
          const error = await apiErrorFrom(response);
          toast.error(`${file.name} was not attached`, {
            description: error.message,
          });
          continue;
        }
        const result = (await response.json()) as UploadResponse;
        setAttachments((current) => [
          ...current,
          { name: result.filename, text: result.text },
        ]);
        if (result.warning) {
          toast.warning(result.filename, { description: result.warning });
        }
      }
    } catch (cause) {
      toast.error("Upload failed", {
        description: cause instanceof Error ? cause.message : String(cause),
      });
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div className="w-full">
      {attachments.length > 0 && (
        <ul className="mb-2 flex flex-wrap gap-1.5">
          {attachments.map((file, index) => (
            <li
              key={`${file.name}-${index}`}
              className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-2 py-1 text-xs"
            >
              <span className="max-w-48 truncate font-medium">{file.name}</span>
              <span className="font-mono text-[10px] text-muted-foreground">
                {file.text.length.toLocaleString()} chars
              </span>
              <button
                type="button"
                aria-label={`Remove ${file.name}`}
                className="rounded p-0.5 text-muted-foreground hover:text-destructive focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                onClick={() =>
                  setAttachments((current) =>
                    current.filter((_, i) => i !== index),
                  )
                }
              >
                <X className="size-3" aria-hidden />
              </button>
            </li>
          ))}
        </ul>
      )}

      <div
        className={cn(
          "flex items-end gap-2 rounded-2xl border border-border bg-card p-2 shadow-sm transition-colors",
          "focus-within:border-brand/50 focus-within:ring-3 focus-within:ring-brand/15",
          disabled && "opacity-60",
        )}
      >
        {uploadsEnabled && (
          <>
            <input
              ref={fileRef}
              type="file"
              multiple
              className="hidden"
              accept=".txt,.md,.markdown,.csv,.json,text/plain,text/markdown,text/csv,application/json"
              onChange={(event) => void upload(event.target.files)}
            />
            <Button
              variant="ghost"
              size="icon-sm"
              type="button"
              aria-label="Attach a text document"
              disabled={disabled || uploading}
              onClick={() => fileRef.current?.click()}
            >
              {uploading ? (
                <Loader2 className="animate-spin" aria-hidden />
              ) : (
                <Paperclip aria-hidden />
              )}
            </Button>
          </>
        )}

        <textarea
          ref={textareaRef}
          value={value}
          rows={1}
          disabled={disabled}
          placeholder={disabled ? (disabledReason ?? placeholder) : placeholder}
          aria-label="Message Bravien"
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            // Enter sends, Shift+Enter breaks the line. IME composition must not
            // be interrupted, or CJK input submits half a word.
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              submit();
            }
          }}
          className="max-h-[280px] min-h-9 flex-1 resize-none bg-transparent px-1.5 py-2 text-[15px] leading-6 outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
        />

        {busy ? (
          <Button
            type="button"
            variant="outline"
            size="icon-sm"
            aria-label="Stop generating"
            onClick={onStop}
          >
            <Square aria-hidden />
          </Button>
        ) : (
          <Button
            type="button"
            size="icon-sm"
            aria-label="Send message"
            disabled={disabled || !value.trim()}
            onClick={submit}
          >
            <ArrowUp aria-hidden />
          </Button>
        )}
      </div>

      <p className="mt-2 px-1 text-[11px] leading-4 text-muted-foreground">
        {disabled && disabledReason ? (
          <span className="text-destructive">{disabledReason}</span>
        ) : (
          <>
            <kbd className="font-mono">Enter</kbd> to send,{" "}
            <kbd className="font-mono">Shift+Enter</kbd> for a new line. Bravien
            runs on this machine and can be wrong — check anything that matters.
          </>
        )}
      </p>
    </div>
  );
}
