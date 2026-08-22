/**
 * The model registry.
 *
 * Models are *discovered* from the local Bravien runtime, not declared here. That
 * is the whole point: the picker lists checkpoints that actually exist and can
 * actually answer, with parameter counts and training steps read from the
 * checkpoint itself (§29, §71).
 *
 * When the runtime is down there are no models. The UI says so rather than
 * offering a name that cannot respond.
 */

import {
  getRuntimeHealth,
  listRuntimeModels,
  toAIModel,
  type RuntimeHealth,
  type RuntimeModelInfo,
} from "@/lib/ai/providers/bravien-local";
import type { AIModel } from "@/types";

/** Cache window. Long enough to spare the runtime, short enough that swapping a
 * checkpoint shows up without a restart. */
const CACHE_TTL_MS = 10_000;

interface RegistrySnapshot {
  models: AIModel[];
  info: RuntimeModelInfo[];
  health: RuntimeHealth;
  fetchedAt: number;
}

let cache: RegistrySnapshot | null = null;
let inflight: Promise<RegistrySnapshot> | null = null;

async function load(): Promise<RegistrySnapshot> {
  const health = await getRuntimeHealth();
  let info: RuntimeModelInfo[] = [];

  if (health.modelLoaded) {
    try {
      info = await listRuntimeModels();
    } catch {
      // Health said a model was loaded but the listing failed. Report zero models
      // rather than inventing one; health carries the reason for the UI.
      info = [];
    }
  }

  return {
    models: info.map(toAIModel),
    info,
    health,
    fetchedAt: Date.now(),
  };
}

async function snapshot(force = false): Promise<RegistrySnapshot> {
  if (!force && cache && Date.now() - cache.fetchedAt < CACHE_TTL_MS) {
    return cache;
  }
  // Collapse concurrent callers onto one request: a page render can ask several
  // times at once, and the runtime serves one generation at a time.
  if (!inflight) {
    inflight = load()
      .then((result) => {
        cache = result;
        return result;
      })
      .finally(() => {
        inflight = null;
      });
  }
  return inflight;
}

/** Models the runtime is serving right now. Empty if none is loaded. */
export async function getModels(): Promise<AIModel[]> {
  return (await snapshot()).models.map((m) => ({ ...m }));
}

export async function getModel(id: string): Promise<AIModel | undefined> {
  const found = (await snapshot()).models.find((m) => m.id === id);
  return found ? { ...found } : undefined;
}

/** Full runtime metadata for a model — architecture, training step, dataset. */
export async function getModelInfo(
  id: string,
): Promise<RuntimeModelInfo | undefined> {
  return (await snapshot()).info.find((m) => m.name === id);
}

/**
 * The model to use when a request names none.
 *
 * Returns null when nothing is loaded, so callers must handle "no model" instead
 * of falling back to a name that does not exist.
 */
export async function getDefaultModelId(): Promise<string | null> {
  const { models } = await snapshot();
  const preferred = process.env.BRAVIEN_DEFAULT_MODEL?.trim();
  if (preferred && models.some((m) => m.id === preferred)) return preferred;
  return models.find((m) => m.isDefault)?.id ?? models[0]?.id ?? null;
}

/** Runtime status for the UI's status indicator. */
export async function getRuntimeStatus(): Promise<RuntimeHealth> {
  return (await snapshot()).health;
}

/** Drop the cache — used after a checkpoint change. */
export function invalidateModelCache(): void {
  cache = null;
}

/** Force a refresh, bypassing the cache. */
export async function refreshModels(): Promise<AIModel[]> {
  return (await snapshot(true)).models.map((m) => ({ ...m }));
}
