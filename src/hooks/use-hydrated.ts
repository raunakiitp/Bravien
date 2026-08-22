"use client";

/**
 * Has the client taken over from the server-rendered HTML?
 *
 * Needed wherever the correct output depends on something only the browser knows
 * — the stored theme, most often. `useSyncExternalStore` is the right primitive
 * for this: `getServerSnapshot` returns false, the client snapshot returns true,
 * and React switches between them at hydration without an effect or a cascading
 * re-render.
 */

import { useSyncExternalStore } from "react";

/** Nothing to subscribe to: the answer changes exactly once, at hydration. */
const subscribe = () => () => {};
const onClient = () => true;
const onServer = () => false;

export function useHydrated(): boolean {
  return useSyncExternalStore(subscribe, onClient, onServer);
}
