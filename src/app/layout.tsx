import type { Metadata, Viewport } from "next";
import { Sora, Geist, Geist_Mono } from "next/font/google";

import { Providers } from "@/components/providers";
import "./globals.css";

/**
 * Fonts are downloaded at build time and served from this machine, so a running
 * Bravien never reaches out for them (§2).
 */
const sans = Geist({
  variable: "--font-bravien-sans",
  subsets: ["latin"],
  display: "swap",
});

const display = Sora({
  variable: "--font-bravien-display",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  display: "swap",
});

const mono = Geist_Mono({
  variable: "--font-bravien-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Bravien",
    template: "%s · Bravien",
  },
  description:
    "Bravien is a language model that runs on your own machine: its own tokenizer, its own weights, its own inference engine. No hosted model service in the path.",
  applicationName: "Bravien",
  // A local tool should not be indexed if it is ever put behind a domain.
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fbfdfe" },
    { media: "(prefers-color-scheme: dark)", color: "#0f1419" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${sans.variable} ${display.variable} ${mono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col bg-background text-foreground">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
