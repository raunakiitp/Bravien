"use client";

/**
 * Client-side providers: colour scheme and toasts.
 *
 * `next-themes` writes the class on `<html>` before paint, which is why the root
 * layout carries `suppressHydrationWarning` — the server cannot know which theme
 * the browser stored.
 */

import { useEffect } from "react";
import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";

export function Providers({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    const handleGlobalError = (event: ErrorEvent) => {
      if (
        event.filename?.startsWith("chrome-extension://") ||
        event.filename?.startsWith("moz-extension://") ||
        event.filename?.startsWith("safari-extension://") ||
        (event.message && typeof event.message === "string" && event.message.includes("M_ID"))
      ) {
        event.stopImmediatePropagation();
        event.preventDefault();
      }
    };
    window.addEventListener("error", handleGlobalError, true);
    return () => window.removeEventListener("error", handleGlobalError, true);
  }, []);

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
      storageKey="bravien-theme"
    >
      {children}
      <Toaster
        position="bottom-center"
        toastOptions={{
          classNames: {
            toast:
              "bg-card text-card-foreground border border-border shadow-lg rounded-xl",
            description: "text-muted-foreground",
          },
        }}
      />
    </ThemeProvider>
  );
}
