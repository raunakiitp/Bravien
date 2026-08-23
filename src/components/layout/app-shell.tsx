"use client";

/**
 * The application shell.
 *
 * One rail, one header, one surface. The header carries the runtime badge because
 * "is there a model behind this box" is the single most useful thing to know
 * before typing, and it should never require opening a menu to find out.
 */

import { useEffect, useState } from "react";
import { Menu, Moon, RefreshCw, Sun } from "lucide-react";
import { useTheme } from "next-themes";

import { Sidebar } from "@/components/layout/sidebar";
import { RuntimeBadge } from "@/components/runtime/status";
import { Button } from "@/components/ui/button";
import type { UseConversationsResult } from "@/hooks/use-conversations";
import { cn } from "@/lib/utils";
import type { RuntimeSnapshot } from "@/types/api";

/**
 * Theme toggle.
 *
 * Both icons are rendered and CSS picks one. `next-themes` sets the `dark` class
 * on `<html>` from a blocking script before first paint, so the right icon is
 * already correct in the server HTML — no mount gate, no wrong icon for a frame.
 */
function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label="Switch between light and dark"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
    >
      <Sun className="hidden dark:block" aria-hidden />
      <Moon className="block dark:hidden" aria-hidden />
    </Button>
  );
}

export interface AppShellProps {
  history: UseConversationsResult;
  snapshot: RuntimeSnapshot | null;
  runtimeLoading: boolean;
  onRefreshRuntime?: () => void;
  activeId?: string | null;
  activeProjectId?: string | null;
  title?: string;
  children: React.ReactNode;
}

export function AppShell({
  history,
  snapshot,
  runtimeLoading,
  onRefreshRuntime,
  activeId = null,
  activeProjectId = null,
  title,
  children,
}: AppShellProps) {
  const [railOpen, setRailOpen] = useState(false);

  useEffect(() => {
    if (!railOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setRailOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [railOpen]);

  return (
    <div className="flex h-dvh min-h-0 overflow-hidden">
      <div className="hidden lg:flex">
        <Sidebar
          history={history}
          activeId={activeId}
          activeProjectId={activeProjectId}
        />
      </div>

      {railOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button
            type="button"
            aria-label="Close history"
            className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            onClick={() => setRailOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 shadow-2xl">
            <Sidebar
              history={history}
              activeId={activeId}
              activeProjectId={activeProjectId}
              onClose={() => setRailOpen(false)}
              // Closed where the navigation happens, rather than by watching the
              // route change from an effect.
              onNavigate={() => setRailOpen(false)}
            />
          </div>
        </div>
      )}

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <header
          className={cn(
            "flex h-13 shrink-0 items-center gap-2 border-b border-border px-3 sm:px-4",
          )}
        >
          <Button
            variant="ghost"
            size="icon-sm"
            className="lg:hidden"
            aria-label="Open conversation history"
            aria-expanded={railOpen}
            onClick={() => setRailOpen(true)}
          >
            <Menu aria-hidden />
          </Button>

          <h1 className="min-w-0 flex-1 truncate font-display text-sm font-medium tracking-tight">
            {title ?? "New conversation"}
          </h1>

          <RuntimeBadge
            snapshot={snapshot}
            loading={runtimeLoading}
            className="hidden sm:inline-flex"
          />
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Re-check the runtime"
            onClick={onRefreshRuntime}
          >
            <RefreshCw className={cn(runtimeLoading && "animate-spin")} aria-hidden />
          </Button>
          <ThemeToggle />
        </header>

        <main className="flex min-h-0 flex-1 flex-col">{children}</main>
      </div>
    </div>
  );
}
