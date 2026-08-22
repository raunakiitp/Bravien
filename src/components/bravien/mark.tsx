/**
 * The Bravien mark.
 *
 * A lattice diamond with a rising path cut through it: the structure is the
 * network, the path is what training carved into it. Drawn from scratch here so
 * Bravien looks like itself and not like a relabelled competitor (§36).
 */

import { cn } from "@/lib/utils";

export function BravienMark({
  className,
  ...props
}: React.ComponentProps<"svg">) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      focusable="false"
      className={cn("size-6", className)}
      {...props}
    >
      <path
        d="M12 2.4 21.6 12 12 21.6 2.4 12Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
        opacity="0.55"
      />
      <path
        d="M7.1 14.6 10.3 9.9l2.9 3.1L16.9 7.7"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="16.9" cy="7.7" r="1.5" fill="currentColor" />
    </svg>
  );
}

/** Mark plus name, for headers and the home screen. */
export function BravienWordmark({
  className,
  markClassName,
  showMark = true,
}: {
  className?: string;
  markClassName?: string;
  showMark?: boolean;
}) {
  return (
    <span className={cn("inline-flex items-center gap-2", className)}>
      {showMark && <BravienMark className={cn("text-brand", markClassName)} />}
      <span className="font-display font-semibold tracking-tight">Bravien</span>
    </span>
  );
}
