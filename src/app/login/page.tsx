"use client";

/**
 * `/login` — only meaningful when accounts are configured.
 *
 * `authConfig.pages.signIn` points here, so this page has to exist even in the
 * common single-machine install where there is nothing to sign in to. In that
 * case it says so and sends you back rather than presenting a form that could
 * never succeed.
 */

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { signIn } from "next-auth/react";
import { ArrowLeft, Loader2 } from "lucide-react";

import { BravienMark, BravienWordmark } from "@/components/bravien/mark";
import { Button } from "@/components/ui/button";
import { useRuntime } from "@/hooks/use-runtime";

export default function LoginPage() {
  const router = useRouter();
  const params = useSearchParams();
  const { snapshot, loading } = useRuntime();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Accounts live in the database. Without one there is nobody to sign in as.
  const accountsPossible = snapshot?.persistence === "database";
  const next = params.get("callbackUrl") ?? "/";

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await signIn("credentials", {
        email,
        password,
        redirect: false,
      });
      if (result?.error) {
        // Deliberately not "no such user" vs "wrong password".
        setError("That email and password do not match an account.");
        return;
      }
      router.push(next);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not reach the sign-in endpoint.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-dvh items-center justify-center px-5 py-12">
      <div className="w-full max-w-sm">
        <div className="flex items-center gap-2.5">
          <BravienMark className="size-7 text-brand" />
          <BravienWordmark className="text-base" />
        </div>

        {loading ? (
          <div className="mt-8 h-44 animate-pulse rounded-2xl bg-muted/60" />
        ) : accountsPossible ? (
          <>
            <h1 className="mt-7 font-display text-2xl font-semibold tracking-tight">
              Sign in
            </h1>
            <p className="mt-1.5 text-sm leading-6 text-muted-foreground">
              Accounts separate saved conversations and memories. Chatting itself
              never requires one.
            </p>

            <form onSubmit={onSubmit} className="mt-6 space-y-3">
              <div>
                <label
                  htmlFor="email"
                  className="text-xs font-medium text-muted-foreground"
                >
                  Email
                </label>
                <input
                  id="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  className="mt-1 h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/40"
                />
              </div>
              <div>
                <label
                  htmlFor="password"
                  className="text-xs font-medium text-muted-foreground"
                >
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className="mt-1 h-10 w-full rounded-lg border border-border bg-background px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/40"
                />
              </div>

              {error && (
                <p role="alert" className="text-xs leading-5 text-destructive">
                  {error}
                </p>
              )}

              <Button type="submit" className="w-full" disabled={busy}>
                {busy && <Loader2 className="animate-spin" aria-hidden />}
                Sign in
              </Button>
            </form>
          </>
        ) : (
          <>
            <h1 className="mt-7 font-display text-2xl font-semibold tracking-tight">
              There is nothing to sign in to
            </h1>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              This Bravien install has no database, so it has no accounts. That is
              the normal setup for one person on one machine — the checkpoint
              answers regardless of who is asking.
            </p>
            <p className="mt-3 font-mono text-xs break-all text-muted-foreground">
              To enable accounts: set DATABASE_URL and AUTH_SECRET in .env, then run
              npm run db:push
            </p>
          </>
        )}

        <Button
          variant="ghost"
          size="sm"
          className="mt-6"
          nativeButton={false}
          render={<Link href="/" />}
        >
          <ArrowLeft aria-hidden /> Back to chat
        </Button>
      </div>
    </div>
  );
}
