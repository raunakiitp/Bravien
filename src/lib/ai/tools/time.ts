import { z } from "zod";
import type { ToolDefinition } from "./base";

const timeSchema = z.object({
  timeZone: z.string().optional(),
});

export const timeTool: ToolDefinition<typeof timeSchema> = {
  name: "get_current_time",
  description: "Get the current local date, time, day of the week, and timezone.",
  schema: timeSchema,
  execute: async ({ timeZone }) => {
    const now = new Date();
    const tz = timeZone || Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
    const dateStr = now.toLocaleDateString("en-US", {
      timeZone: tz,
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    const timeStr = now.toLocaleTimeString("en-US", {
      timeZone: tz,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
    });
    const iso = now.toISOString();

    return {
      success: true,
      data: { date: dateStr, time: timeStr, timeZone: tz, iso },
      formattedOutput: `Current Date & Time: ${dateStr}, ${timeStr} (${tz})`,
    };
  },
};
