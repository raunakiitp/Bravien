"use client";

/**
 * Runtime and checkpoint status.
 *
 * Every number here is read from the loaded checkpoint via `/api/runtime` — the
 * parameter count is counted, the training step is the step it stopped at, the
 * dataset is the one it saw. Nothing is rounded up for effect, and a checkpoint
 * trained on the synthetic seed corpus says so, because a user deciding whether
 * to trust an answer needs that more than a reassuring badge (§29, §71, §72).
 */

import { AlertTriangle, CircleSlash, Cpu, Zap } from "lucide-react";

import { cn } from "@/lib/utils";
import type { RuntimeModelInfo, RuntimeSnapshot } from "@/types/api";

/** True when the training data was the generated seed corpus, not real text. */
export function isSyntheticCheckpoint(model: RuntimeModelInfo): boolean {
  const sources = model.training?.dataset?.sources;
  if (!Array.isArray(sources)) return false;
  return sources.some(
    (source) => typeof source === "string" && source.includes("synthetic"),
  );
}

export function formatParameters(count: number): string {
  if (count >= 1_000_000_000) return `${(count / 1_000_000_000).toFixed(2)}B`;
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(2)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}K`;
  return String(count);
}

type Tone = "ok" | "warn" | "down";

function toneOf(snapshot: RuntimeSnapshot | null): Tone {
  if (!snapshot || snapshot.runtime.status === "unreachable") return "down";
  if (!snapshot.runtime.modelLoaded) return "warn";
  return "ok";
}

const dotClass: Record<Tone, string> = {
  ok: "bg-brand",
  warn: "bg-chart-4",
  down: "bg-destructive",
};

/** Compact status for the header. */
export function RuntimeBadge({
  snapshot,
  loading,
  className,
}: {
  snapshot: RuntimeSnapshot | null;
  loading: boolean;
  className?: string;
}) {
  const tone = toneOf(snapshot);
  const model = snapshot?.activeModel;

  const label = loading
    ? "Checking runtime"
    : tone === "down"
      ? "Runtime offline"
      : tone === "warn"
        ? "No checkpoint loaded"
        : (model?.name ?? snapshot?.runtime.model ?? "Checkpoint loaded");

  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border border-border bg-card/70 px-2.5 py-1 text-xs",
        className,
      )}
      title={
        tone === "ok" && model
          ? `${formatParameters(model.parameters)} parameters · step ${model.training.step.toLocaleString()} · ${model.precision} on ${model.device}`
          : (snapshot?.runtime.error ?? undefined)
      }
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          dotClass[tone],
          loading && "animate-pulse",
        )}
        aria-hidden
      />
      <span className="font-medium">{label}</span>
      {tone === "ok" && model && (
        <span className="font-mono text-[10px] text-muted-foreground">
          {formatParameters(model.parameters)} · step{" "}
          {model.training.step.toLocaleString()}
        </span>
      )}
      <span className="sr-only">
        {tone === "ok"
          ? "Bravien runtime is serving a checkpoint."
          : tone === "warn"
            ? "The Bravien runtime is running but no checkpoint is loaded."
            : "The Bravien runtime is not reachable."}
      </span>
    </span>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="truncate font-mono text-xs text-foreground">{value}</dd>
    </div>
  );
}

/** What is actually loaded — architecture, training, tokenizer, dataset. */
export function ModelCard({
  model,
  className,
}: {
  model: RuntimeModelInfo;
  className?: string;
}) {
  const synthetic = isSyntheticCheckpoint(model);
  const trainLoss = model.training.metrics?.train_loss;
  const evalLoss = model.training.metrics?.eval_loss;
  const dataset = model.training.dataset as {
    tokens?: number;
    documents?: number;
    sources?: unknown;
  };

  return (
    <div className={cn("rounded-2xl border border-border bg-card", className)}>
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <Cpu className="size-4 text-brand" aria-hidden />
        <h2 className="font-display text-sm font-semibold tracking-tight">
          {model.name}
        </h2>
        <span className="font-mono text-[10px] text-muted-foreground">
          v{model.version}
        </span>
      </div>

      {synthetic && (
        <div className="flex items-start gap-2.5 border-b border-border bg-chart-4/10 px-4 py-3">
          <AlertTriangle
            className="mt-0.5 size-4 shrink-0 text-chart-4"
            aria-hidden
          />
          <p className="text-xs leading-5 text-foreground/85">
            This checkpoint was trained on Bravien&apos;s{" "}
            <strong className="font-semibold">synthetic seed corpus</strong>, not
            on real text. It proves the pipeline end to end — tokenizer, training,
            checkpoint, inference — and it will not answer questions usefully.
            Point <code className="font-mono">prepare_data.py --local-dir</code> at
            a real corpus and retrain for a model worth talking to.
          </p>
        </div>
      )}

      <div className="grid gap-x-8 px-4 py-3 sm:grid-cols-2">
        <dl className="divide-y divide-border/60">
          <Row
            label="Parameters"
            value={`${formatParameters(model.parameters)} (${model.parameters.toLocaleString()})`}
          />
          <Row label="Architecture" value={model.architecture} />
          <Row
            label="Layers / heads"
            value={`${model.layers} / ${model.heads} (${model.kv_heads} KV)`}
          />
          <Row label="Hidden size" value={model.hidden_size} />
          <Row
            label="Context"
            value={`${model.context_length.toLocaleString()} tokens`}
          />
          <Row
            label="Norm / position / act"
            value={`${model.norm} · ${model.position_encoding} · ${model.activation}`}
          />
        </dl>

        <dl className="divide-y divide-border/60">
          <Row
            label="Stage / step"
            value={`${model.training.stage} · ${model.training.step.toLocaleString()}`}
          />
          <Row
            label="Tokens seen"
            value={model.training.tokens_seen.toLocaleString()}
          />
          {typeof trainLoss === "number" && (
            <Row label="Train loss" value={trainLoss.toFixed(4)} />
          )}
          {typeof evalLoss === "number" && (
            <Row label="Eval loss" value={evalLoss.toFixed(4)} />
          )}
          <Row
            label="Vocabulary"
            value={`${model.tokenizer.vocab_size.toLocaleString()} tokens`}
          />
          <Row label="Precision / device" value={`${model.precision} · ${model.device}`} />
          {typeof dataset.tokens === "number" && (
            <Row
              label="Corpus"
              value={`${dataset.tokens.toLocaleString()} tokens${
                typeof dataset.documents === "number"
                  ? ` · ${dataset.documents.toLocaleString()} docs`
                  : ""
              }`}
            />
          )}
        </dl>
      </div>

      {model.checkpoint && (
        <p className="border-t border-border px-4 py-2.5 font-mono text-[10px] break-all text-muted-foreground">
          {model.checkpoint}
        </p>
      )}
    </div>
  );
}

/** Shown when there is nothing to talk to, with the command that fixes it. */
export function RuntimeDown({
  snapshot,
  className,
}: {
  snapshot: RuntimeSnapshot | null;
  className?: string;
}) {
  const unreachable = !snapshot || snapshot.runtime.status === "unreachable";

  return (
    <div
      className={cn(
        "rounded-2xl border border-destructive/30 bg-destructive/5 p-5",
        className,
      )}
      role="status"
    >
      <div className="flex items-start gap-3">
        {unreachable ? (
          <CircleSlash className="mt-0.5 size-5 shrink-0 text-destructive" aria-hidden />
        ) : (
          <Zap className="mt-0.5 size-5 shrink-0 text-chart-4" aria-hidden />
        )}
        <div className="min-w-0">
          <h2 className="font-display text-base font-semibold tracking-tight">
            {unreachable
              ? "The Bravien runtime is not running"
              : "No checkpoint is loaded"}
          </h2>
          <p className="mt-1.5 text-sm leading-6 text-foreground/80">
            {unreachable
              ? "Bravien answers from a checkpoint served on this machine. Nothing else can answer for it, so there is nothing to send a message to yet."
              : "The runtime is up but has no weights. It needs a trained checkpoint before it can generate anything."}
          </p>

          <div className="mt-3 overflow-x-auto rounded-lg border border-border bg-background p-3">
            <pre className="font-mono text-xs leading-6 text-foreground/90">
              {unreachable
                ? `python scripts/serve.py --checkpoint checkpoints/bravien`
                : `python scripts/train_tokenizer.py\npython scripts/prepare_data.py\npython scripts/pretrain.py --preset tiny --max-steps 1500\npython scripts/serve.py --checkpoint checkpoints/bravien`}
            </pre>
          </div>

          {snapshot?.runtime.error && (
            <p className="mt-2.5 font-mono text-[11px] break-all text-muted-foreground">
              {snapshot.runtime.error}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
