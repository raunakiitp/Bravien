"use client";

/**
 * Settings — what this install actually is.
 *
 * Deliberately not a wall of toggles. Most of Bravien's behaviour is decided by
 * which checkpoint is served and what is in the environment, so this page reports
 * those and is explicit about which capabilities have no implementation behind
 * them. A switch that does nothing is worse than no switch (§72).
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useTheme } from "next-themes";
import {
  ArrowLeft,
  Check,
  Database,
  HardDrive,
  Minus,
  Monitor,
  Moon,
  Sun,
} from "lucide-react";

import { BravienWordmark } from "@/components/bravien/mark";
import { ModelCard, RuntimeDown } from "@/components/runtime/status";
import { Button } from "@/components/ui/button";
import { useHydrated } from "@/hooks/use-hydrated";
import { useRuntime } from "@/hooks/use-runtime";
import { cn } from "@/lib/utils";
import type { FeatureFlags } from "@/types";

/** Capabilities with no implementation. Listed so their absence is legible. */
const NOT_IMPLEMENTED: ReadonlyArray<{
  key: keyof FeatureFlags;
  label: string;
  why: string;
}> = [
  {
    key: "vision",
    label: "Image understanding",
    why: "Bravien's architecture is text-only — there is no vision encoder in the model.",
  },
  {
    key: "voice",
    label: "Speech in or out",
    why: "No audio model is trained or served.",
  },
  {
    key: "image_generation",
    label: "Image generation",
    why: "Bravien is a language model; no image decoder exists.",
  },
  {
    key: "code_execution",
    label: "Running code",
    why: "Held back on purpose until there is a real sandbox — the model will not be given a shell on this machine.",
  },
];

function Section({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-t border-border py-7 first:border-t-0 first:pt-0">
      <h2 className="font-display text-base font-semibold tracking-tight">
        {title}
      </h2>
      {description && (
        <p className="mt-1 max-w-2xl text-sm leading-6 text-muted-foreground">
          {description}
        </p>
      )}
      <div className="mt-4">{children}</div>
    </section>
  );
}

function ThemeChoice() {
  const { theme, setTheme } = useTheme();
  // "system" is not expressible in CSS, so the highlight genuinely needs to know
  // whether the client has taken over yet.
  const hydrated = useHydrated();

  const options = [
    { value: "light", label: "Light", icon: Sun },
    { value: "dark", label: "Dark", icon: Moon },
    { value: "system", label: "System", icon: Monitor },
  ] as const;

  return (
    <div className="inline-flex rounded-xl border border-border p-1">
      {options.map(({ value, label, icon: Icon }) => {
        const active = hydrated && theme === value;
        return (
          <button
            key={value}
            type="button"
            aria-pressed={active}
            onClick={() => setTheme(value)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors",
              "focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
              active
                ? "bg-secondary text-secondary-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <Icon className="size-3.5" aria-hidden />
            {label}
          </button>
        );
      })}
    </div>
  );
}

function FlagRow({
  label,
  on,
  detail,
}: {
  label: string;
  on: boolean;
  detail: string;
}) {
  return (
    <li className="flex items-start gap-3 py-2.5">
      <span
        className={cn(
          "mt-0.5 flex size-4.5 shrink-0 items-center justify-center rounded-full",
          on ? "bg-brand/15 text-brand" : "bg-muted text-muted-foreground",
        )}
        aria-hidden
      >
        {on ? <Check className="size-3" /> : <Minus className="size-3" />}
      </span>
      <div className="min-w-0">
        <p className="text-sm font-medium">
          {label}
          <span className="ml-2 text-xs font-normal text-muted-foreground">
            {on ? "on" : "off"}
          </span>
        </p>
        <p className="text-xs leading-5 text-muted-foreground">{detail}</p>
      </div>
    </li>
  );
}

function MemoryManagement() {
  const [memories, setMemories] = useState<
    Array<{ id: string; type: string; content: string; updatedAt: string }>
  >([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [busy, setBusy] = useState(false);

  const fetchMemories = useCallback(async () => {
    try {
      const res = await fetch("/api/memories");
      if (res.ok) {
        const data = await res.json();
        setMemories(data.items || []);
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchMemories();
  }, [fetchMemories]);

  async function handleDelete(id: string) {
    setBusy(true);
    try {
      await fetch(`/api/memories/${id}`, { method: "DELETE" });
      setMemories((prev) => prev.filter((m) => m.id !== id));
    } finally {
      setBusy(false);
    }
  }

  async function handleClearAll() {
    if (!window.confirm("Clear all stored memories? This cannot be undone.")) return;
    setBusy(true);
    try {
      await fetch("/api/memories", { method: "DELETE" });
      setMemories([]);
    } finally {
      setBusy(false);
    }
  }

  const filtered = filter.trim()
    ? memories.filter((m) =>
        m.content.toLowerCase().includes(filter.trim().toLowerCase()),
      )
    : memories;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <input
          type="text"
          placeholder="Filter memories..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="h-8 w-64 rounded-lg border border-border bg-background px-2.5 text-xs outline-none focus-visible:border-ring focus-visible:ring-2"
        />
        {memories.length > 0 && (
          <Button
            variant="ghost"
            size="xs"
            disabled={busy}
            onClick={handleClearAll}
            className="text-destructive hover:bg-destructive/10"
          >
            Clear all
          </Button>
        )}
      </div>

      {loading ? (
        <div className="h-16 animate-pulse rounded-xl bg-muted/40" />
      ) : filtered.length === 0 ? (
        <p className="rounded-xl border border-dashed border-border py-6 text-center text-xs text-muted-foreground">
          {filter.trim() ? "No matching memories." : "No memories stored yet."}
        </p>
      ) : (
        <ul className="divide-y divide-border rounded-xl border border-border bg-card">
          {filtered.map((m) => (
            <li key={m.id} className="flex items-center justify-between p-3 text-xs">
              <div className="min-w-0 pr-3">
                <div className="flex items-center gap-1.5">
                  <span className="rounded bg-brand/10 px-1.5 py-0.5 font-mono text-[10px] text-brand">
                    {m.type}
                  </span>
                  <span className="text-[10px] text-muted-foreground">
                    {new Date(m.updatedAt).toLocaleDateString()}
                  </span>
                </div>
                <p className="mt-1 font-medium text-foreground">{m.content}</p>
              </div>
              <Button
                variant="ghost"
                size="xs"
                disabled={busy}
                onClick={() => handleDelete(m.id)}
                className="text-muted-foreground hover:text-destructive"
              >
                Delete
              </Button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function SettingsPage() {
  const { snapshot, loading, refresh } = useRuntime();
  const model = snapshot?.activeModel ?? null;
  const features = snapshot?.features;
  const database = snapshot?.persistence === "database";

  return (
    <div className="h-dvh overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl px-5 py-8 sm:px-8">
        <div className="flex items-center justify-between gap-4">
          <BravienWordmark className="text-sm" />
          <Button
            variant="ghost"
            size="sm"
            nativeButton={false}
            render={<Link href="/" />}
          >
            <ArrowLeft aria-hidden /> Back to chat
          </Button>
        </div>

        <h1 className="mt-6 font-display text-3xl font-semibold tracking-tight">
          Settings & Identity
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">
          Bravien is a local, private AI assistant. Active models serve as runtime execution engines under the Bravien identity.
        </p>

        <div className="mt-8">
          <Section
            title="Assistant Identity & Model Engine"
            description="Bravien identity with runtime engine metadata."
          >
            {model ? (
              <ModelCard model={model} />
            ) : loading ? (
              <div className="h-40 animate-pulse rounded-2xl bg-muted/60" />
            ) : (
              <RuntimeDown snapshot={snapshot} />
            )}
            <Button
              variant="outline"
              size="sm"
              className="mt-3"
              onClick={() => void refresh()}
            >
              Re-check the runtime
            </Button>
          </Section>

          <Section
            title="Persistent Memories"
            description="Cross-session preferences, project context, and facts retained by Bravien."
          >
            <MemoryManagement />
          </Section>

          <Section title="Appearance">
            <ThemeChoice />
          </Section>

          <Section
            title="Storage"
            description="Where conversations go after you close the tab."
          >
            <div
              className={cn(
                "flex items-start gap-3 rounded-2xl border p-4",
                database
                  ? "border-border bg-card"
                  : "border-chart-4/30 bg-chart-4/5",
              )}
            >
              {database ? (
                <Database className="mt-0.5 size-4 shrink-0 text-brand" aria-hidden />
              ) : (
                <HardDrive
                  className="mt-0.5 size-4 shrink-0 text-chart-4"
                  aria-hidden
                />
              )}
              <div className="min-w-0 text-sm leading-6">
                {database ? (
                  <>
                    <p className="font-medium">Saved to a database</p>
                    <p className="text-muted-foreground">
                      Conversations, projects, files, and memories are written to PostgreSQL.
                    </p>
                  </>
                ) : (
                  <>
                    <p className="font-medium">Nothing is being saved</p>
                    <p className="text-muted-foreground">
                      No database is configured, so conversations live only in this tab.
                    </p>
                  </>
                )}
              </div>
            </div>
          </Section>

          <Section
            title="Capabilities"
            description="What exists in this build. Items marked off have no code behind them — turning a flag on would not create the feature."
          >
            {features ? (
              <ul className="divide-y divide-border/60">
                <FlagRow
                  label="Text documents"
                  on={features.file_uploads}
                  detail="Plain text, Markdown, CSV and JSON are read on the server and their text is sent with your message."
                />
                <FlagRow
                  label="Memory"
                  on={features.memory}
                  detail="Facts you ask Bravien to remember, stored per account. Needs a database."
                />
                <FlagRow
                  label="Web search"
                  on={features.web_search}
                  detail="Off by default. Bravien answers from its weights alone; nothing leaves this machine."
                />
                {NOT_IMPLEMENTED.map(({ key, label, why }) => (
                  <FlagRow
                    key={key}
                    label={label}
                    on={features[key]}
                    detail={why}
                  />
                ))}
              </ul>
            ) : (
              <div className="h-32 animate-pulse rounded-xl bg-muted/60" />
            )}
          </Section>

          <Section
            title="Where answers come from"
            description="For the avoidance of doubt."
          >
            <div className="rounded-2xl border border-border bg-card p-4">
              <p className="font-mono text-xs leading-6 break-words text-foreground/85">
                browser → /api/chat → Bravien inference engine → checkpoint →
                Bravien neural network
              </p>
              <p className="mt-3 text-sm leading-6 text-muted-foreground">
                No hosted model API is part of that path. Unplug the network and a
                served checkpoint keeps answering; the only thing that stops
                working is anything that was never local to begin with.
              </p>
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}
