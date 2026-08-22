"use client";

/**
 * Live runtime status.
 *
 * Everything the UI says about the model comes from here, and everything here
 * comes from `/api/runtime`, which reads the loaded checkpoint. When the runtime
 * is down there is no model list and no parameter count to show — the UI says
 * that plainly instead of rendering a plausible-looking placeholder (§71, §72).
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { apiFetch, type RuntimeSnapshot } from "@/types/api";

/** Slow poll: the answer only changes when someone starts or stops the runtime. */
const POLL_INTERVAL_MS = 15_000;
/** Faster poll while it is down, so a just-started runtime appears quickly. */
const DOWN_POLL_INTERVAL_MS = 4_000;

export interface UseRuntimeResult {
  snapshot: RuntimeSnapshot | null;
  /** True only until the first answer arrives; a refresh does not blank the UI. */
  loading: boolean;
  /** The status request itself failed — the dev server, not the runtime. */
  error: string | null;
  ready: boolean;
  refresh: () => Promise<void>;
}

export function useRuntime(): UseRuntimeResult {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  /**
   * Fetch status.
   *
   * Written as a promise chain rather than `async`/`await` on purpose: the state
   * updates land in callbacks, so mounting does not schedule a synchronous
   * re-render inside the effect flush.
   */
  const load = useCallback((fresh: boolean) => {
    return apiFetch<RuntimeSnapshot>(
      fresh ? "/api/runtime?refresh=1" : "/api/runtime",
    )
      .then((data) => {
        if (!mounted.current) return;
        setSnapshot(data);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (!mounted.current) return;
        setError(cause instanceof Error ? cause.message : String(cause));
      })
      .finally(() => {
        if (mounted.current) setLoading(false);
      });
  }, []);

  useEffect(() => {
    mounted.current = true;
    void load(false);
    return () => {
      mounted.current = false;
    };
  }, [load]);

  // Re-arm on every status change so the interval matches the current state
  // rather than the one at mount.
  const modelLoaded = snapshot?.runtime.modelLoaded ?? false;
  useEffect(() => {
    const interval = modelLoaded ? POLL_INTERVAL_MS : DOWN_POLL_INTERVAL_MS;
    const timer = setInterval(() => void load(true), interval);
    return () => clearInterval(timer);
  }, [load, modelLoaded]);

  const refresh = useCallback(async () => {
    await load(true);
  }, [load]);
  return {
    snapshot,
    loading,
    error,
    ready: modelLoaded && (snapshot?.models.length ?? 0) > 0,
    refresh,
  };
}
