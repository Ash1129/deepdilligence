import { cn } from "@/lib/utils";
import type { Rating } from "@/lib/api";

export function ConfidenceBadge({
  value,
  className,
}: {
  value: number;
  className?: string;
}) {
  const pct = Math.round(value * 100);
  const tone =
    value >= 0.7
      ? "bg-success/15 text-success border-success/30"
      : value >= 0.5
        ? "bg-warning/15 text-warning border-warning/30"
        : "bg-danger/15 text-danger border-danger/30";
  return (
    <span
      className={cn(
        "tnum inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium",
        tone,
        className,
      )}
    >
      {pct}% conf
    </span>
  );
}

const RATING_STYLES: Record<Rating, string> = {
  "STRONG BUY": "bg-[oklch(0.45_0.16_145)] text-white",
  BUY: "bg-[oklch(0.55_0.18_145)] text-white",
  HOLD: "bg-[oklch(0.55_0.16_75)] text-white",
  SELL: "bg-[oklch(0.45_0.20_27)] text-white",
  "STRONG SELL": "bg-[oklch(0.35_0.20_27)] text-white",
};

export function RatingBadge({ rating }: { rating: Rating }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded px-2 py-0.5 text-[11px] font-bold tracking-wide uppercase",
        RATING_STYLES[rating],
      )}
    >
      {rating}
    </span>
  );
}