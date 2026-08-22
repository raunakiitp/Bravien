/**
 * GET /api/runtime — what Bravien can actually do right now.
 *
 * The UI renders its status indicator, model picker and model card from this. All
 * of it is measured: the runtime reports the loaded checkpoint's real parameter
 * count, training step and context length, and this route forwards them without
 * embellishment (§29, §71).
 */

import { getModels, getRuntimeStatus, refreshModels } from "@/lib/ai/models";
import { getModelInfo } from "@/lib/ai/models";
import { isDatabaseReachable } from "@/lib/db/available";
import { getFeatureFlags } from "@/lib/features";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const fresh = url.searchParams.get("refresh") === "1";

  const [health, models, persistence] = await Promise.all([
    getRuntimeStatus(),
    fresh ? refreshModels() : getModels(),
    isDatabaseReachable(),
  ]);

  const active = health.model ? await getModelInfo(health.model) : undefined;

  return Response.json(
    {
      runtime: health,
      models,
      // Full architecture/training detail for the model card. Absent when no
      // checkpoint is loaded — there is nothing truthful to show.
      activeModel: active ?? null,
      features: getFeatureFlags(),
      persistence: persistence ? "database" : "ephemeral",
    },
    {
      headers: {
        // Status must never be served stale: the whole point is telling the user
        // whether the model is up right now.
        "Cache-Control": "no-store, must-revalidate",
      },
    },
  );
}
