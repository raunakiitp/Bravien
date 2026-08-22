"use client";

/**
 * Client-side providers: colour scheme and toasts.
 *
 * `next-themes` writes the class on `<html>` before paint, which is why the root
 * layout carries `suppressHydrationWarning` — the server cannot know which theme
 * the browser stored.
 */

import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";

export function Providers({ children }: { children: React.ReactNode }) {
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
