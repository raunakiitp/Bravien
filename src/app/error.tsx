"use client";

/**
 * Root error boundary.
 *
 * Shows the actual error, because the person reading it is almost always the
 * person running the machine that produced it — a generic apology would cost them
 * the one useful detail.
 */

import { useEffect } from "react";
import Link from "next/link";
import { RotateCcw } from "lucide-react";

import { BravienMark } from "@/components/bravien/mark";
import { Button } from "@/components/ui/button";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("[bravien] unhandled error", error);
  }, [error]);

  return (
    <div className="flex min-h-dvh items-center justify-center px-5 py-12">
      <div className="w-full max-w-md">
        <BravienMark className="size-8 text-brand" />
        <h1 className="mt-4 font-display text-2xl font-semibold tracking-tight">
          Something in the interface broke
        </h1>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          This is a bug in Bravien&apos;s web layer, not a refusal by the model.
        </p>

        <pre className="mt-4 overflow-x-auto rounded-xl border border-border bg-card p-3 font-mono text-xs leading-5 whitespace-pre-wrap text-foreground/85">
          {error.message || "No message was attached to the error."}
        </pre>
        {error.digest && (
          <p className="mt-2 font-mono text-[11px] text-muted-foreground">
            digest {error.digest}
          </p>
        )}

        <div className="mt-5 flex flex-wrap gap-2">
          <Button onClick={reset}>
            <RotateCcw aria-hidden /> Try again
          </Button>
          <Button variant="outline" nativeButton={false} render={<Link href="/" />}>
            Start a new conversation
          </Button>
        </div>
      </div>
    </div>
  );
}
