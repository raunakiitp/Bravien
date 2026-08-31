"use client";

/**
 * Markdown rendering for assistant turns.
 *
 * Model output is untrusted text. Raw HTML is not enabled (no `rehype-raw`), so a
 * checkpoint that emits `<script>` gets an escaped string rather than a script
 * tag, and react-markdown's default URL transform drops `javascript:` hrefs. Both
 * are load-bearing: the text on screen came from a generator, not from a person
 * with good intentions (§53).
 */

import { memo, useRef, useState } from "react";
import { Check, Copy, Terminal } from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { cn } from "@/lib/utils";

/** Read `language-xyz` off the wrapped `<code>` for the block's label. */
function languageOf(node: React.ReactNode): string | null {
  if (
    typeof node === "object" &&
    node !== null &&
    "props" in node &&
    typeof (node as { props?: unknown }).props === "object"
  ) {
    const className = (node as { props: { className?: unknown } }).props
      .className;
    if (typeof className === "string") {
      const match = /language-([\w+-]+)/.exec(className);
      if (match) return match[1];
    }
  }
  return null;
}

/**
 * A fenced code block with a copy button.
 *
 * The text is read back out of the DOM rather than reconstructed from the React
 * tree: `rehype-highlight` has already split it into spans, and re-assembling
 * those risks copying something subtly different from what is displayed.
 */
function CodeBlock({
  children,
  className,
  ...props
}: React.ComponentProps<"pre">) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);
  const language = languageOf(children);

  const copy = async () => {
    const text = ref.current?.textContent ?? "";
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard access can be refused (insecure origin, denied permission).
    }
  };

  return (
    <div className="group/code relative my-4 rounded-xl border border-border/90 bg-card overflow-hidden shadow-xs">
      <div className="flex items-center justify-between border-b border-border/80 bg-muted/50 px-3.5 py-1.5">
        <div className="flex items-center gap-1.5">
          <Terminal className="size-3.5 text-muted-foreground" aria-hidden />
          <span className="font-mono text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            {language ?? "code"}
          </span>
        </div>
        <button
          type="button"
          onClick={copy}
          className="inline-flex items-center gap-1 rounded-md border border-border/60 bg-background/80 px-2 py-0.5 text-[11px] text-muted-foreground transition-all hover:bg-background hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none cursor-pointer"
          aria-label={copied ? "Copied code" : "Copy code"}
        >
          {copied ? (
            <>
              <Check className="size-3 text-brand" aria-hidden />
              <span className="text-brand font-medium">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="size-3" aria-hidden />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre
        ref={ref}
        className={cn("mt-0! p-4 font-mono text-[13px] leading-relaxed overflow-x-auto bg-transparent", className)}
        {...props}
      >
        {children}
      </pre>
    </div>
  );
}

const components: Components = {
  pre: CodeBlock,
  a({ href, children, ...props }) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noreferrer noopener"
        className="text-brand underline underline-offset-2 hover:text-brand/80 transition-colors"
        {...props}
      >
        {children}
      </a>
    );
  },
  // Tables need a scroll container or a wide one breaks the message column.
  table({ children, ...props }) {
    return (
      <div className="my-4 overflow-x-auto rounded-lg border border-border/80">
        <table className="w-full text-sm" {...props}>
          {children}
        </table>
      </div>
    );
  },
};

const remarkPlugins = [remarkGfm, remarkMath];
const rehypePlugins = [
  rehypeKatex,
  // `ignoreMissing` matters while streaming: a half-written fence can name a
  // language that does not exist yet, and throwing would blank the turn.
  [rehypeHighlight, { detect: true, ignoreMissing: true }] as const,
];

/**
 * Memoised on `content`: a streaming turn re-renders on every delta, and
 * re-parsing unchanged siblings for each one is the difference between smooth
 * and stuttering output.
 */
export const Markdown = memo(function Markdown({
  content,
  className,
}: {
  content: string;
  className?: string;
}) {
  return (
    <div className={cn("prose-bravien leading-7 text-[15px]", className)}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        rehypePlugins={rehypePlugins as any}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
