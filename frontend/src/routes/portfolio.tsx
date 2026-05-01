import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Header } from "@/components/Header";
import { ConfidenceBadge, RatingBadge } from "@/components/badges";
import { postJSON, type Portfolio, type PortfolioHolding } from "@/lib/api";
import {
  Loader2,
  ChevronDown,
  TrendingUp,
  TrendingDown,
  DollarSign,
  BarChart3,
  PieChart,
  AlertTriangle,
  Briefcase,
} from "lucide-react";
import { cn } from "@/lib/utils";
import heroPortfolio from "@/assets/hero-portfolio.jpg";

export const Route = createFileRoute("/portfolio")({
  component: PortfolioPage,
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmt$(n: number) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(n);
}

function fmtPct(n: number | null | undefined, plus = true) {
  if (n == null) return "—";
  const sign = plus && n > 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

const RATING_COLOR: Record<string, string> = {
  "STRONG BUY": "text-emerald-400",
  "BUY":        "text-green-400",
};

// ── Page ─────────────────────────────────────────────────────────────────────

function PortfolioPage() {
  const [amount, setAmount]           = useState<string>("");
  const [maxPos, setMaxPos]           = useState(10);
  const [maxPosPct, setMaxPosPct]     = useState(30);
  const [strongOnly, setStrongOnly]   = useState(false);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [portfolio, setPortfolio]     = useState<Portfolio | null>(null);

  async function build() {
    const amt = parseFloat(amount.replace(/,/g, ""));
    if (!amt || amt <= 0) { setError("Enter a valid investment amount."); return; }
    setLoading(true);
    setError(null);
    try {
      const data = await postJSON<Portfolio>("/api/portfolio/build", {
        amount: amt,
        max_positions: maxPos,
        max_position_pct: maxPosPct,
        strong_buy_only: strongOnly,
      });
      setPortfolio(data);
    } catch (e: unknown) {
      setError((e as Error).message ?? "Failed to build portfolio. Generate a weekly report first.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative min-h-screen">
      {/* Hero */}
      <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 -z-0 h-[110vh] overflow-hidden">
        <div className="absolute inset-0 bg-cover bg-center" style={{ backgroundImage: `url(${heroPortfolio})` }} />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,_oklch(0.22_0.04_55_/_0.18)_0%,_oklch(0.22_0.06_55_/_0.35)_45%,_oklch(0.20_0.05_50_/_0.85)_75%,_var(--background)_100%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_oklch(0.78_0.14_70_/_0.18),_transparent_60%)]" />
      </div>

      <Header subtle />

      <main className="relative z-10 mx-auto max-w-7xl px-6 py-8">
        <div className="grid gap-6 lg:grid-cols-[300px_1fr]">

          {/* ── Sidebar ── */}
          <aside className="lg:sticky lg:top-20 lg:self-start space-y-4">
            <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-5 backdrop-blur-sm">
              <div className="mb-1 flex items-center gap-2">
                <Briefcase className="h-4 w-4 text-[oklch(0.78_0.14_70)]" />
                <h2 className="text-sm font-semibold uppercase tracking-wider text-white/80">Portfolio Builder</h2>
              </div>
              <p className="mb-5 text-xs text-white/60">
                AI-ranked picks from this week's report, weighted by confidence.
              </p>

              {/* Amount */}
              <label className="mb-3 block">
                <span className="mb-1 block text-xs font-medium text-white/75">Investment Amount (USD)</span>
                <div className="relative">
                  <DollarSign className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-white/40" />
                  <input
                    type="text"
                    value={amount}
                    onChange={(e) => setAmount(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && build()}
                    placeholder="10,000"
                    className="tnum w-full rounded-md border border-white/15 bg-[oklch(0.18_0.02_55_/_0.70)] py-2 pl-8 pr-3 text-sm text-white outline-none placeholder:text-white/40 focus:border-white/40"
                  />
                </div>
              </label>

              {/* Max positions */}
              <label className="mb-3 block">
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-xs font-medium text-white/75">Max Holdings</span>
                  <span className="tnum text-xs text-[oklch(0.78_0.14_70)]">{maxPos}</span>
                </div>
                <input
                  type="range" min={3} max={20} value={maxPos}
                  onChange={(e) => setMaxPos(Number(e.target.value))}
                  className="w-full accent-[oklch(0.72_0.14_65)]"
                />
                <div className="mt-0.5 flex justify-between text-[10px] text-white/40">
                  <span>3</span><span>20</span>
                </div>
              </label>

              {/* Max position size */}
              <label className="mb-4 block">
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-xs font-medium text-white/75">Max Position Size</span>
                  <span className="tnum text-xs text-[oklch(0.78_0.14_70)]">{maxPosPct}%</span>
                </div>
                <input
                  type="range" min={10} max={50} step={5} value={maxPosPct}
                  onChange={(e) => setMaxPosPct(Number(e.target.value))}
                  className="w-full accent-[oklch(0.72_0.14_65)]"
                />
                <div className="mt-0.5 flex justify-between text-[10px] text-white/40">
                  <span>10%</span><span>50%</span>
                </div>
              </label>

              {/* Rating filter */}
              <div className="mb-5">
                <span className="mb-2 block text-xs font-medium text-white/75">Minimum Rating</span>
                <div className="grid grid-cols-2 gap-2">
                  {[false, true].map((val) => (
                    <button
                      key={String(val)}
                      onClick={() => setStrongOnly(val)}
                      className={cn(
                        "rounded-md border px-3 py-2 text-xs font-medium transition",
                        strongOnly === val
                          ? "border-[oklch(0.75_0.14_65)] bg-[oklch(0.55_0.15_55_/_0.40)] text-white"
                          : "border-white/10 text-white/50 hover:border-white/20 hover:text-white/80"
                      )}
                    >
                      {val ? "⭐ Strong Buy only" : "✅ Buy & above"}
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={build}
                disabled={loading || !amount.trim()}
                className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-[oklch(0.55_0.15_55)] px-3 py-2.5 text-sm font-semibold text-white shadow-lg shadow-black/20 transition hover:bg-[oklch(0.62_0.17_58)] disabled:opacity-50"
              >
                {loading ? <><Loader2 className="h-4 w-4 animate-spin" /> Building…</> : <><Briefcase className="h-4 w-4" /> Build Portfolio</>}
              </button>
            </div>

            {/* Macro note */}
            {portfolio?.macro_commentary && (
              <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-4 backdrop-blur-sm">
                <p className="mb-1 text-xs font-semibold uppercase tracking-wider text-white/70">Macro View</p>
                <p className="text-xs leading-relaxed text-white/65">{portfolio.macro_commentary}</p>
              </div>
            )}
          </aside>

          {/* ── Main ── */}
          <section className="space-y-5">

            {error && (
              <div className="flex items-start gap-3 rounded-lg border border-danger/40 bg-[oklch(0.20_0.03_55_/_0.75)] p-4 text-sm text-danger backdrop-blur-sm">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                  <div className="font-semibold">Unable to build portfolio</div>
                  <div className="text-danger/80">{error}</div>
                </div>
              </div>
            )}

            {!portfolio && !loading && !error && <EmptyState />}

            {portfolio && (
              <>
                {/* ── Summary cards ── */}
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <SummaryCard label="Total Invested" value={fmt$(portfolio.total_invested)} icon={<DollarSign className="h-4 w-4" />} />
                  <SummaryCard label="Holdings" value={String(portfolio.num_holdings)} icon={<Briefcase className="h-4 w-4" />} />
                  <SummaryCard label="Cash Remainder" value={fmt$(portfolio.cash_remainder)} icon={<BarChart3 className="h-4 w-4" />} />
                  <SummaryCard
                    label="Avg Confidence"
                    value={`${(portfolio.avg_confidence * 100).toFixed(0)}%`}
                    icon={<PieChart className="h-4 w-4" />}
                  />
                </div>

                {/* ── S&P 500 benchmark ── */}
                {portfolio.sp500?.latest_close && (
                  <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-5 backdrop-blur-sm">
                    <div className="mb-3 flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-[oklch(0.78_0.14_70)]" />
                      <h3 className="text-sm font-semibold uppercase tracking-wider text-white/80">S&amp;P 500 Benchmark</h3>
                      <span className="tnum ml-auto text-sm font-semibold text-white">{portfolio.sp500.latest_close.toLocaleString()}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      {[
                        { label: "1 Week",   val: portfolio.sp500.return_1w  },
                        { label: "1 Month",  val: portfolio.sp500.return_1m  },
                        { label: "3 Months", val: portfolio.sp500.return_3m  },
                        { label: "YTD",      val: portfolio.sp500.return_ytd },
                      ].map(({ label, val }) => (
                        <div key={label} className="rounded-lg border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] px-4 py-3 text-center">
                          <div className="text-xs text-white/50">{label}</div>
                          <div className={cn("tnum mt-1 text-base font-bold",
                            val == null ? "text-white/40" : val >= 0 ? "text-emerald-400" : "text-red-400"
                          )}>
                            {fmtPct(val)}
                          </div>
                        </div>
                      ))}
                    </div>
                    <p className="mt-3 text-xs text-white/45">
                      Beat this benchmark by combining AI-ranked conviction picks with disciplined position sizing.
                      Week of {portfolio.week_of} · {portfolio.universe_size} stocks analysed.
                    </p>
                  </div>
                )}

                {/* ── Holdings table ── */}
                <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] backdrop-blur-sm">
                  <div className="border-b border-white/10 px-5 py-4">
                    <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-white/80">
                      <Briefcase className="h-4 w-4 text-[oklch(0.78_0.14_70)]" />
                      Portfolio Holdings
                      <span className="ml-auto text-xs font-normal text-white/50">
                        {portfolio.strong_buy_count} Strong Buy · {portfolio.num_holdings - portfolio.strong_buy_count} Buy
                      </span>
                    </h3>
                  </div>
                  <div className="divide-y divide-white/5">
                    {portfolio.holdings.map((h, i) => (
                      <HoldingRow key={h.ticker} holding={h} index={i} />
                    ))}
                  </div>
                </div>

                {/* ── Sector breakdown ── */}
                <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-5 backdrop-blur-sm">
                  <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-white/80">
                    <PieChart className="h-4 w-4 text-[oklch(0.78_0.14_70)]" /> Sector Allocation
                  </h3>
                  <div className="space-y-3">
                    {Object.entries(portfolio.sector_breakdown).map(([sector, pct]) => (
                      <div key={sector}>
                        <div className="mb-1 flex items-center justify-between text-xs">
                          <span className="text-white/80">{sector}</span>
                          <span className="tnum text-white/60">{pct.toFixed(1)}%</span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-[oklch(0.15_0.02_55_/_0.80)]">
                          <div
                            className="h-full rounded-full bg-[oklch(0.62_0.17_58)]"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="rounded-xl border border-dashed border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-12 text-center backdrop-blur-sm">
      <Briefcase className="mx-auto mb-3 h-8 w-8 text-[oklch(0.78_0.14_70)]" />
      <h3 className="text-lg font-semibold">Build your AI portfolio</h3>
      <p className="mt-1 text-sm text-white/60">
        Enter an investment amount and click Build Portfolio. Requires a Weekly Rankings report.
      </p>
    </div>
  );
}

function SummaryCard({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-4 backdrop-blur-sm">
      <div className="mb-2 flex items-center gap-1.5 text-[oklch(0.78_0.14_70)]">{icon}</div>
      <div className="tnum text-xl font-bold text-white">{value}</div>
      <div className="mt-0.5 text-xs text-white/55">{label}</div>
    </div>
  );
}

function HoldingRow({ holding: h, index }: { holding: PortfolioHolding; index: number }) {
  const [open, setOpen] = useState(false);

  return (
    <div>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-3 px-5 py-3.5 text-left transition hover:bg-white/5"
      >
        {/* Rank */}
        <span className="tnum w-5 shrink-0 text-center text-xs text-white/35">#{index + 1}</span>

        {/* Ticker + Company */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="tnum font-semibold text-white">{h.ticker}</span>
            <span className={cn("text-xs font-semibold", RATING_COLOR[h.rating] ?? "text-white/60")}>
              {h.rating}
            </span>
          </div>
          <div className="truncate text-xs text-white/55">{h.company} · {h.sector}</div>
        </div>

        {/* Weight + Amount */}
        <div className="tnum hidden text-right sm:block">
          <div className="text-sm font-semibold text-white">{h.weight_pct.toFixed(1)}%</div>
          <div className="text-xs text-white/55">{fmt$(h.dollar_amount)}</div>
        </div>

        {/* Shares + Price */}
        <div className="tnum hidden text-right md:block">
          <div className="text-sm text-white">{h.shares.toFixed(h.shares >= 1 ? 2 : 4)} sh</div>
          <div className="text-xs text-white/55">@ {fmt$(h.current_price)}</div>
        </div>

        {/* Confidence */}
        <div className="shrink-0">
          <ConfidenceBadge value={h.confidence} />
        </div>

        <ChevronDown className={cn("h-4 w-4 shrink-0 text-white/30 transition", open && "rotate-180")} />
      </button>

      {open && (
        <div className="border-t border-white/10 bg-[oklch(0.20_0.03_55_/_0.70)] px-5 py-4 backdrop-blur-sm">
          {/* Mobile row */}
          <div className="mb-3 flex gap-4 sm:hidden">
            <div className="tnum text-center">
              <div className="text-sm font-semibold text-white">{h.weight_pct.toFixed(1)}%</div>
              <div className="text-xs text-white/55">{fmt$(h.dollar_amount)}</div>
            </div>
            <div className="tnum text-center">
              <div className="text-sm text-white">{h.shares.toFixed(4)} sh</div>
              <div className="text-xs text-white/55">@ {fmt$(h.current_price)}</div>
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-md border border-white/10 bg-[oklch(0.18_0.02_55_/_0.75)] p-3">
              <div className="mb-1 flex items-center gap-1 text-xs font-semibold text-emerald-400">
                <TrendingUp className="h-3 w-3" /> Bull Case
              </div>
              <p className="text-xs leading-relaxed text-white/80">{h.bull_case}</p>
            </div>
            <div className="rounded-md border border-white/10 bg-[oklch(0.18_0.02_55_/_0.75)] p-3">
              <div className="mb-1 flex items-center gap-1 text-xs font-semibold text-red-400">
                <TrendingDown className="h-3 w-3" /> Bear Case
              </div>
              <p className="text-xs leading-relaxed text-white/80">{h.bear_case}</p>
            </div>
          </div>
          {h.rationale && (
            <div className="mt-3 rounded-md border border-white/10 bg-[oklch(0.18_0.02_55_/_0.75)] p-3">
              <div className="mb-1 text-xs font-semibold text-white/60">Rationale</div>
              <p className="text-xs leading-relaxed text-white/80">{h.rationale}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
