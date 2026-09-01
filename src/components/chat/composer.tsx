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
import { ArrowUp, Loader2, Mic, MicOff, Paperclip, Square, X } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import type { ChatAttachment } from "@/hooks/use-chat";
import { BrowserSpeechInputProvider, isSpeechRecognitionSupported } from "@/lib/voice/speech";
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
  placeholder = "Message Bravien (Enter to send, Shift+Enter for newline)",
  autoFocus = false,
}: ComposerProps) {
  const [value, setValue] = useState("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [uploading, setUploading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const speechProviderRef = useRef<BrowserSpeechInputProvider | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    speechProviderRef.current = new BrowserSpeechInputProvider();
    return () => {
      speechProviderRef.current?.stop();
    };
  }, []);

  function toggleVoice() {
    if (!isSpeechRecognitionSupported()) {
      toast.error("Speech recognition is not supported in this browser.");
      return;
    }

    if (isListening) {
      speechProviderRef.current?.stop();
      setIsListening(false);
    } else {
      const started = speechProviderRef.current?.start({
        onTranscript: (text) => {
          setValue((prev) => (prev ? `${prev} ${text}` : text));
        },
        onError: (err) => {
          setIsListening(false);
          toast.error(`Voice error: ${err}`);
        },
        onEnd: () => {
          setIsListening(false);
        },
      });
      if (started) setIsListening(true);
    }
  }

  // Grow with content up to maximum ceiling
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
    if (isListening) {
      speechProviderRef.current?.stop();
      setIsListening(false);
    }
    onSend(text, attachments);
    setValue("");
    setAttachments([]);
    requestAnimationFrame(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
        textareaRef.current.focus();
      }
    });
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
              className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-2.5 py-1 text-xs shadow-2xs"
            >
              <span className="max-w-48 truncate font-medium text-foreground">{file.name}</span>
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
          "flex items-end gap-1.5 rounded-2xl border border-border/90 bg-card p-2 shadow-xs transition-all",
          "focus-within:border-brand/60 focus-within:ring-3 focus-within:ring-brand/15",
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
              className="rounded-xl text-muted-foreground hover:text-foreground"
              disabled={disabled || uploading}
              onClick={() => fileRef.current?.click()}
            >
              {uploading ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : (
                <Paperclip className="size-4" aria-hidden />
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
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              submit();
            }
          }}
          className="max-h-[280px] min-h-9 flex-1 resize-none bg-transparent px-2 py-2 text-[15px] leading-6 outline-none placeholder:text-muted-foreground/80 disabled:cursor-not-allowed"
        />

        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label={isListening ? "Stop microphone" : "Speak message"}
          disabled={disabled}
          onClick={toggleVoice}
          className={cn(
            "rounded-xl text-muted-foreground hover:text-foreground",
            isListening && "text-brand bg-brand/10 animate-pulse",
          )}
        >
          {isListening ? <MicOff className="size-4" aria-hidden /> : <Mic className="size-4" aria-hidden />}
        </Button>

        {busy ? (
          <Button
            type="button"
            variant="outline"
            size="icon-sm"
            aria-label="Stop generating"
            className="rounded-xl border-border/90 bg-muted/60 text-foreground hover:bg-destructive/10 hover:text-destructive hover:border-destructive/30"
            onClick={onStop}
          >
            <Square className="size-3.5 fill-current" aria-hidden />
          </Button>
        ) : (
          <Button
            type="button"
            size="icon-sm"
            aria-label="Send message"
            className="rounded-xl bg-brand text-brand-foreground hover:bg-brand/90 transition-transform active:scale-95"
            disabled={disabled || !value.trim()}
            onClick={submit}
          >
            <ArrowUp className="size-4 stroke-[2.5]" aria-hidden />
          </Button>
        )}
      </div>

      <div className="mt-2 flex items-center justify-between px-1 text-[11px] leading-4 text-muted-foreground">
        {disabled && disabledReason ? (
          <span className="text-destructive font-medium">{disabledReason}</span>
        ) : (
          <span>
            Bravien-v1 offline intelligence · Private & local on your machine
          </span>
        )}
      </div>
    </div>
  );
}
