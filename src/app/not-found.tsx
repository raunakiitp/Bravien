/** 404. */

import Link from "next/link";

import { BravienMark } from "@/components/bravien/mark";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="flex min-h-dvh items-center justify-center px-5 py-12">
      <div className="w-full max-w-md">
        <BravienMark className="size-8 text-brand" />
        <h1 className="mt-4 font-display text-2xl font-semibold tracking-tight">
          Nothing lives at this address
        </h1>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          The page you asked for does not exist. If you followed a link to a
          conversation, it may have been deleted.
        </p>
        <Button
          className="mt-5"
          nativeButton={false}
          render={<Link href="/" />}
        >
          Start a new conversation
        </Button>
      </div>
    </div>
  );
}
