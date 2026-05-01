import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Header } from "@/components/Header";
import heroWeekly from "@/assets/hero-weekly.jpg";
import { ConfidenceBadge, RatingBadge } from "@/components/badges";
import {
  getJSON,
  streamSSE,
  type StockRating,
  type UniverseResponse,
  type WeeklyReport,
  type WeeklyReportRequest,
} from "@/lib/api";
import {
  Loader2,
  Play,
  CheckCircle2,
  XCircle,
  ChevronDown,
  Sparkles,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/weekly")({
  component: WeeklyPage,
});

type ScreenerCriteria = WeeklyReportRequest["screener_criteria"];

interface ProgressStock {
  ticker: string;
  company?: string;
  status: "pending" | "running" | "done" | "error";
  message?: string;
  sections?: number;
  confidence?: number;
}

function WeeklyPage() {
  const [universe, setUniverse] = useState<UniverseResponse | null>(null);
  const [styles, setStyles] = useState<string[]>([]);
  const [sectors, setSectors] = useState<string[]>([]);
  const [topN, setTopN] = useState(10);
  const [useScreener, setUseScreener] = useState(false);
  const [criteria, setCriteria] = useState<ScreenerCriteria>("Price Change");

  const [report, setReport] = useState<WeeklyReport | null>(null);
  const [history, setHistory] = useState<WeeklyReport[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [watchlist, setWatchlist] = useState<ProgressStock[] | null>(null);
  const [ranking, setRanking] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getJSON<UniverseResponse>("/api/universe")
      .then((u) => {
        setUniverse(u);
      })
      .catch(() => {});
    getJSON<WeeklyReport>("/api/weekly-report")
      .then(setReport)
      .catch(() => {});
    getJSON<WeeklyReport[]>("/api/weekly-report/all")
      .then((all) => setHistory(all ?? []))
      .catch(() => {});
  }, []);

  async function generate() {
    if (running) return;
    setRunning(true);
    setError(null);
    setWatchlist(null);
    setRanking(false);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamSSE(
        "/api/weekly-report/generate",
        {
          styles,
          sectors,
          top_n: topN,
          use_screener: useScreener,
          screener_criteria: criteria,
        } satisfies WeeklyReportRequest,
        (ev) => {
          if (ev.type === "watchlist_start") {
            setWatchlist([]);
          } else if (ev.type === "watchlist_complete") {
            setWatchlist(
              (ev.stocks ?? []).map((s: any) => ({
                ticker: s.ticker,
                company: s.company,
                status: "pending" as const,
              })),
            );
          } else if (ev.type === "stock_start") {
            setWatchlist((prev) =>
              (prev ?? []).map((s) =>
                s.ticker === ev.ticker ? { ...s, status: "running" } : s,
              ),
            );
          } else if (ev.type === "stock_complete") {
            setWatchlist((prev) =>
              (prev ?? []).map((s) =>
                s.ticker === ev.ticker
                  ? {
                      ...s,
                      status: "done",
                      sections: ev.sections,
                      confidence: ev.confidence,
                    }
                  : s,
              ),
            );
          } else if (ev.type === "stock_error") {
            setWatchlist((prev) =>
              (prev ?? []).map((s) =>
                s.ticker === ev.ticker
                  ? { ...s, status: "error", message: ev.message }
                  : s,
              ),
            );
          } else if (ev.type === "ranking_start") {
            setRanking(true);
          } else if (ev.type === "complete") {
            setReport(ev.report);
            setRanking(false);
          } else if (ev.type === "error") {
            setError(ev.message ?? "Generation failed");
          }
        },
        ctrl.signal,
      );
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  }

  useEffect(() => () => abortRef.current?.abort(), []);

  return (
    <div className="relative min-h-screen">
      {/* Hero background — same skyscraper as landing page */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 -z-0 h-[110vh] overflow-hidden"
      >
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${heroWeekly})` }}
        />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,_oklch(0.22_0.06_55_/_0.55)_0%,_oklch(0.20_0.06_52_/_0.72)_45%,_oklch(0.18_0.05_50_/_0.92)_75%,_var(--background)_100%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_oklch(0.65_0.14_58_/_0.35),_transparent_60%)]" />
      </div>
      <Header subtle />
      <main className="relative z-10 mx-auto max-w-7xl px-6 py-8">
        <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
          <aside className="lg:sticky lg:top-20 lg:self-start">
            <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-5 backdrop-blur-sm">
              <h2 className="mb-1 text-sm font-semibold uppercase tracking-wider text-white/80">
                Preferences
              </h2>
              <p className="mb-4 text-xs text-white/60">
                Configure your weekly portfolio universe.
              </p>

              <MultiSelect
                label="Investment Style"
                options={universe?.all_styles ?? []}
                value={styles}
                onChange={setStyles}
              />
              <MultiSelect
                label="Sectors"
                options={universe?.all_sectors ?? []}
                value={sectors}
                onChange={setSectors}
              />

              <div className="mb-4">
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span className="font-medium text-white/75">
                    Number of stocks
                  </span>
                  <span className="tnum font-semibold">{topN}</span>
                </div>
                <input
                  type="range"
                  min={3}
                  max={20}
                  value={topN}
                  onChange={(e) => setTopN(Number(e.target.value))}
                  className="w-full accent-[oklch(0.72_0.14_65)]"
                />
              </div>

              <label className="mb-3 flex items-center justify-between rounded-md border border-white/10 bg-[oklch(0.20_0.03_55_/_0.50)] px-3 py-2 text-sm">
                <span>Use live S&P 500 screener</span>
                <input
                  type="checkbox"
                  checked={useScreener}
                  onChange={(e) => setUseScreener(e.target.checked)}
                  className="h-4 w-4 accent-[oklch(0.72_0.14_65)]"
                />
              </label>

              {useScreener && (
                <label className="mb-4 block">
                  <span className="mb-1 block text-xs font-medium text-white/75">
                    Rank by
                  </span>
                  <select
                    value={criteria}
                    onChange={(e) =>
                      setCriteria(e.target.value as ScreenerCriteria)
                    }
                    className="w-full rounded-md border border-white/10 bg-[oklch(0.18_0.02_55_/_0.70)] px-3 py-2 text-sm text-white outline-none focus:border-white/30"
                  >
                    <option>Price Change</option>
                    <option>Volume</option>
                    <option>Price Change + Volume</option>
                  </select>
                </label>
              )}

              <button
                onClick={generate}
                disabled={running}
                className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-[oklch(0.55_0.15_55)] px-3 py-2.5 text-sm font-semibold text-white shadow-lg shadow-black/20 transition hover:bg-[oklch(0.62_0.17_58)] disabled:opacity-50"
              >
                {running ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" /> Generating…
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" /> Generate Weekly Report
                  </>
                )}
              </button>
            </div>
          </aside>

          <section className="space-y-6">
            {(running || watchlist) && (
              <ProgressView
                watchlist={watchlist}
                ranking={ranking}
                done={!!report && !running}
              />
            )}

            {error && (
              <div className="flex items-start gap-3 rounded-lg border border-danger/40 bg-[oklch(0.20_0.03_55_/_0.75)] backdrop-blur-sm p-4 text-sm text-danger">
                <AlertTriangle className="mt-0.5 h-4 w-4" />
                <div>{error}</div>
              </div>
            )}

            {report ? (
              <ReportView report={report} />
            ) : (
              !running && (
                <div className="rounded-xl border border-dashed border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] backdrop-blur-sm p-12 text-center">
                  <Sparkles className="mx-auto mb-3 h-8 w-8 text-[oklch(0.78_0.14_70)]" />
                  <h3 className="text-lg font-semibold">No report yet</h3>
                  <p className="mt-1 text-sm text-white/60">
                    Configure preferences and generate your first weekly report.
                  </p>
                </div>
              )
            )}

            {history.length > 0 && <HistoryView reports={history} />}
          </section>
        </div>
      </main>
    </div>
  );
}

function MultiSelect({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: string[];
  value: string[];
  onChange: (v: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  function toggle(o: string) {
    onChange(value.includes(o) ? value.filter((v) => v !== o) : [...value, o]);
  }

  const summary =
    options.length === 0
      ? "Loading…"
      : value.length === 0
        ? `Choose ${label}`
        : value.length === options.length
          ? `All ${label}`
          : value.length <= 2
            ? value.join(", ")
            : `${value.length} selected`;

  return (
    <div className="relative mb-4" ref={ref}>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-xs font-medium text-white/75">{label}</span>
        <button
          onClick={() => onChange(value.length === options.length ? [] : [...options])}
          className="text-[11px] text-white/60 hover:text-white hover:underline"
        >
          {value.length === options.length ? "Clear" : "All"}
        </button>
      </div>

      {/* Trigger */}
      <button
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "flex w-full items-center justify-between rounded-md border border-white/15 bg-[oklch(0.18_0.02_55_/_0.70)] px-3 py-2 text-sm transition hover:border-white/30",
          value.length === 0 ? "text-white/40" : "text-white/90",
        )}
      >
        <span>{summary}</span>
        <ChevronDown className={cn("h-4 w-4 text-white/50 transition", open && "rotate-180")} />
      </button>

      {/* Dropdown panel */}
      {open && (
        <div className="absolute z-50 mt-1 w-full overflow-hidden rounded-md border border-white/10 bg-[oklch(0.22_0.04_55_/_0.95)] shadow-2xl backdrop-blur-md">
          <div className="max-h-52 overflow-y-auto p-1">
            {options.map((o) => {
              const active = value.includes(o);
              return (
                <button
                  key={o}
                  onClick={() => toggle(o)}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded px-2.5 py-1.5 text-left text-sm transition",
                    active
                      ? "bg-[oklch(0.55_0.15_55_/_0.40)] text-white"
                      : "text-white/65 hover:bg-white/5 hover:text-white",
                  )}
                >
                  <span
                    className={cn(
                      "flex h-4 w-4 shrink-0 items-center justify-center rounded border text-[10px] font-bold",
                      active
                        ? "border-[oklch(0.75_0.14_65)] bg-[oklch(0.55_0.15_55)] text-white"
                        : "border-white/25",
                    )}
                  >
                    {active && "✓"}
                  </span>
                  {o}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function ProgressView({
  watchlist,
  ranking,
  done,
}: {
  watchlist: ProgressStock[] | null;
  ranking: boolean;
  done: boolean;
}) {
  return (
    <div className="space-y-3 rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-5 backdrop-blur-sm">
      <div className="flex items-center gap-2 text-sm font-semibold">
        {watchlist === null ? (
          <Loader2 className="h-4 w-4 animate-spin text-[oklch(0.78_0.14_70)]" />
        ) : (
          <CheckCircle2 className="h-4 w-4 text-success" />
        )}
        🔍 Building watchlist…
      </div>
      {watchlist && watchlist.length > 0 && (
        <ul className="grid gap-1.5 sm:grid-cols-2">
          {watchlist.map((s) => (
            <li
              key={s.ticker}
              className="flex items-center justify-between rounded-md border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] px-3 py-2 text-sm"
            >
              <span className="flex items-center gap-2">
                {s.status === "done" && (
                  <CheckCircle2 className="h-4 w-4 text-success" />
                )}
                {s.status === "running" && (
                  <Loader2 className="h-4 w-4 animate-spin text-[oklch(0.78_0.14_70)]" />
                )}
                {s.status === "error" && <XCircle className="h-4 w-4 text-danger" />}
                {s.status === "pending" && (
                  <span className="h-4 w-4 rounded-full border border-white/20" />
                )}
                <span className="tnum font-semibold">{s.ticker}</span>
                <span className="truncate text-xs text-white/60">
                  {s.company}
                </span>
              </span>
              {s.status === "done" && (
                <ConfidenceBadge value={s.confidence ?? 0} />
              )}
              {s.status === "error" && (
                <span className="truncate text-xs text-danger">{s.message}</span>
              )}
            </li>
          ))}
        </ul>
      )}
      <div className="flex items-center gap-2 text-sm">
        {done ? (
          <CheckCircle2 className="h-4 w-4 text-success" />
        ) : ranking ? (
          <Loader2 className="h-4 w-4 animate-spin text-[oklch(0.78_0.14_70)]" />
        ) : (
          <span className="h-4 w-4 rounded-full border border-white/20" />
        )}
        <span className={cn(!ranking && !done && "text-white/60")}>
          🧠 Ranking via portfolio manager LLM…
        </span>
      </div>
    </div>
  );
}

function ReportView({ report }: { report: WeeklyReport }) {
  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-5 backdrop-blur-sm">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <div className="text-xs uppercase tracking-wider text-white/70">
              Weekly Report
            </div>
            <h2 className="text-2xl font-bold">📈 Week of {report.week_of}</h2>
          </div>
          <div className="tnum text-xs text-white/60">
            {new Date(report.generated_at).toLocaleString()} · {report.model_used} ·{" "}
            {report.universe_size} stocks
          </div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-xl border border-success/40 bg-[oklch(0.20_0.03_55_/_0.75)] backdrop-blur-sm p-4">
          <div className="mb-1 text-xs font-semibold uppercase text-success">
            ✅ Top Picks
          </div>
          <div className="tnum text-base font-semibold text-foreground">
            {report.top_picks.join(", ") || "—"}
          </div>
        </div>
        {report.avoid.length > 0 && (
          <div className="rounded-xl border border-danger/40 bg-[oklch(0.20_0.03_55_/_0.75)] backdrop-blur-sm p-4">
            <div className="mb-1 text-xs font-semibold uppercase text-danger">
              ❌ Avoid
            </div>
            <div className="tnum text-base font-semibold text-foreground">
              {report.avoid.join(", ")}
            </div>
          </div>
        )}
      </div>

      {report.macro_commentary && (
        <div className="rounded-xl border border-white/15 bg-[oklch(0.20_0.03_55_/_0.75)] backdrop-blur-sm p-4 text-sm leading-relaxed text-foreground/90">
          <div className="mb-1 text-xs font-semibold uppercase text-[oklch(0.78_0.14_70)]">
            Macro commentary
          </div>
          {report.macro_commentary}
        </div>
      )}

      {Object.keys(report.sector_views).length > 0 && (
        <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-5 backdrop-blur-sm">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-white/70">
            Sector views
          </h3>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(report.sector_views).map(([sector, view]) => (
              <SectorTile key={sector} sector={sector} view={view} />
            ))}
          </div>
        </div>
      )}

      <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] backdrop-blur-sm">
        <div className="border-b border-white/10 p-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-white/70">
            Rankings ({report.ratings.length})
          </h3>
        </div>
        <ul className="divide-y divide-border">
          {[...report.ratings]
            .sort((a, b) => a.rank - b.rank)
            .map((r) => (
              <RankingRow key={r.ticker} r={r} />
            ))}
        </ul>
      </div>
    </div>
  );
}

function SectorTile({ sector, view }: { sector: string; view: string }) {
  const v = view.toLowerCase();
  const isBull = v.includes("bull");
  const isBear = v.includes("bear");
  const Icon = isBull ? TrendingUp : isBear ? TrendingDown : Minus;
  const tone = isBull ? "text-success" : isBear ? "text-danger" : "text-white/60";
  return (
    <div className="flex items-center justify-between rounded-md border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] px-3 py-2 text-sm">
      <span className="text-foreground/90">{sector}</span>
      <span className={cn("flex items-center gap-1.5 text-xs font-semibold", tone)}>
        <Icon className="h-3.5 w-3.5" />
        {view}
      </span>
    </div>
  );
}

function RankingRow({ r }: { r: StockRating }) {
  const [open, setOpen] = useState(false);
  return (
    <li>
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left text-sm transition hover:bg-[oklch(0.26_0.05_55_/_0.40)]"
      >
        <div className="flex min-w-0 items-center gap-3">
          <span className="tnum w-8 text-right text-xs text-white/60">
            #{r.rank}
          </span>
          <span className="tnum w-14 font-bold">{r.ticker}</span>
          <span className="truncate text-foreground/90">{r.company_name}</span>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <RatingBadge rating={r.rating} />
          <span className="tnum hidden text-xs text-white/60 sm:inline">
            weight {r.suggested_weight_pct.toFixed(1)}%
          </span>
          <ConfidenceBadge value={r.confidence} />
          <ChevronDown
            className={cn("h-4 w-4 transition", open && "rotate-180")}
          />
        </div>
      </button>
      {open && (
        <div className="border-t border-white/10 bg-[oklch(0.20_0.03_55_/_0.70)] backdrop-blur-sm px-4 py-4">
          <div className="mb-3 flex flex-wrap items-center gap-3 text-xs text-white/60">
            <span className="rounded-full border border-white/10 bg-[oklch(0.18_0.02_55_/_0.60)] px-2 py-0.5">
              {r.sector}
            </span>
            <span className="tnum">Weight: {r.suggested_weight_pct.toFixed(1)}%</span>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-success/30 bg-[oklch(0.18_0.02_55_/_0.75)] p-3 text-sm">
              <div className="mb-1 text-xs font-semibold text-success">🐂 Bull case</div>
              <p className="text-foreground/90">{r.bull_case}</p>
            </div>
            <div className="rounded-md border border-danger/30 bg-[oklch(0.18_0.02_55_/_0.75)] p-3 text-sm">
              <div className="mb-1 text-xs font-semibold text-danger">🐻 Bear case</div>
              <p className="text-foreground/90">{r.bear_case}</p>
            </div>
          </div>
          <div className="mt-3 rounded-md border border-white/10 bg-[oklch(0.18_0.02_55_/_0.75)] p-3 text-sm">
            <div className="mb-1 text-xs font-semibold uppercase text-white/60">
              Rationale
            </div>
            <p className="text-foreground/90">{r.rationale}</p>
          </div>
        </div>
      )}
    </li>
  );
}

function HistoryView({ reports }: { reports: WeeklyReport[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] backdrop-blur-sm">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-5 py-4 text-sm font-semibold"
      >
        <span>📚 Historical reports ({reports.length})</span>
        <ChevronDown className={cn("h-4 w-4 transition", open && "rotate-180")} />
      </button>
      {open && (
        <ul className="divide-y divide-white/10 border-t border-white/10">
          {reports.map((r, i) => (
            <li
              key={i}
              className="flex items-center justify-between px-5 py-3 text-sm"
            >
              <span>Week of {r.week_of}</span>
              <span className="tnum text-xs text-white/60">
                {r.ratings.length} stocks ·{" "}
                {new Date(r.generated_at).toLocaleDateString()}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}