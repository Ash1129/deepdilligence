import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { z } from "zod";
import { Header } from "@/components/Header";
import { ConfidenceBadge } from "@/components/badges";
import {
  AGENT_META,
  streamSSE,
  type InvestmentMemo,
} from "@/lib/api";
import {
  CheckCircle2,
  Loader2,
  Sparkles,
  Play,
  Download,
  ChevronDown,
  AlertTriangle,
  TrendingUp,
} from "lucide-react";
import { cn } from "@/lib/utils";
import heroAnalyze from "@/assets/hero-analyze.avif";

const searchSchema = z.object({
  company: z.string().optional(),
  ticker: z.string().optional(),
});

export const Route = createFileRoute("/analyze")({
  validateSearch: searchSchema,
  component: AnalyzePage,
});

type AgentStatus = "idle" | "running" | "done";
interface AgentRow {
  agent: string;
  status: AgentStatus;
  findings?: number;
  confidence?: number;
}

function AnalyzePage() {
  const search = Route.useSearch();
  const [company, setCompany] = useState(search.company ?? "");
  const [ticker, setTicker] = useState(search.ticker ?? "");
  const [running, setRunning] = useState(false);
  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [synthesizing, setSynthesizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [memo, setMemo] = useState<InvestmentMemo | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Auto-fill from query (and re-trigger when changed)
  useEffect(() => {
    if (search.company) setCompany(search.company);
    if (search.ticker) setTicker(search.ticker);
  }, [search.company, search.ticker]);

  async function run() {
    if (!company.trim() || running) return;
    setRunning(true);
    setError(null);
    setMemo(null);
    setSynthesizing(false);
    setAgents([]);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamSSE(
        "/api/analyze",
        { company: company.trim(), ticker: ticker.trim() || undefined },
        (ev) => {
          if (ev.type === "agent_start") {
            setAgents((prev) =>
              prev.find((a) => a.agent === ev.agent)
                ? prev
                : [...prev, { agent: ev.agent, status: "running" }],
            );
          } else if (ev.type === "agent_complete") {
            setAgents((prev) =>
              prev.map((a) =>
                a.agent === ev.agent
                  ? {
                      ...a,
                      status: "done",
                      findings: ev.findings,
                      confidence: ev.confidence,
                    }
                  : a,
              ),
            );
          } else if (ev.type === "synthesis_start") {
            setSynthesizing(true);
          } else if (ev.type === "complete") {
            setMemo(ev.memo);
            setSynthesizing(false);
          } else if (ev.type === "error") {
            setError(ev.message ?? "Pipeline error");
          }
        },
        ctrl.signal,
      );
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message);
      }
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
          style={{ backgroundImage: `url(${heroAnalyze})` }}
        />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,_oklch(0.22_0.04_55_/_0.18)_0%,_oklch(0.22_0.06_55_/_0.35)_45%,_oklch(0.20_0.05_50_/_0.85)_75%,_var(--background)_100%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_oklch(0.78_0.14_70_/_0.18),_transparent_60%)]" />
      </div>
      <Header subtle />
      <main className="relative z-10 mx-auto max-w-7xl px-6 py-8">
        <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
          {/* Sidebar */}
          <aside className="lg:sticky lg:top-20 lg:self-start">
            <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-5 backdrop-blur-sm">
              <h2 className="mb-1 text-sm font-semibold uppercase tracking-wider text-white/80">
                Single company
              </h2>
              <p className="mb-4 text-xs text-white/60">
                Run all five specialist agents in parallel.
              </p>
              <label className="mb-3 block">
                <span className="mb-1 block text-xs font-medium text-white/75">
                  Company name
                </span>
                <input
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                  placeholder="Apple Inc"
                  className="w-full rounded-md border border-white/15 bg-[oklch(0.18_0.02_55_/_0.70)] px-3 py-2 text-sm text-white outline-none placeholder:text-white/40 focus:border-white/40"
                />
              </label>
              <label className="mb-4 block">
                <span className="mb-1 block text-xs font-medium text-white/75">
                  Ticker (optional)
                </span>
                <input
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  placeholder="AAPL"
                  className="tnum w-full rounded-md border border-white/15 bg-[oklch(0.18_0.02_55_/_0.70)] px-3 py-2 text-sm uppercase text-white outline-none placeholder:text-white/40 focus:border-white/40"
                />
              </label>
              <button
                onClick={run}
                disabled={running || !company.trim()}
                className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-[oklch(0.55_0.15_55)] px-3 py-2.5 text-sm font-semibold text-white shadow-lg shadow-black/20 transition hover:bg-[oklch(0.62_0.17_58)] disabled:opacity-50"
              >
                {running ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" /> Running…
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" /> Run Analysis
                  </>
                )}
              </button>
            </div>
          </aside>

          {/* Right side */}
          <section className="space-y-6">
            {(running || agents.length > 0 || synthesizing) && (
              <ProgressPanel
                agents={agents}
                synthesizing={synthesizing}
                done={!!memo}
              />
            )}

            {error && (
              <div className="flex items-start gap-3 rounded-lg border border-danger/40 bg-[oklch(0.20_0.03_55_/_0.75)] backdrop-blur-sm p-4 text-sm text-danger">
                <AlertTriangle className="mt-0.5 h-4 w-4" />
                <div>
                  <div className="font-semibold">Analysis failed</div>
                  <div className="text-danger/80">{error}</div>
                </div>
              </div>
            )}

            {memo && <MemoView memo={memo} />}

            {!memo && !running && agents.length === 0 && <EmptyState />}
          </section>
        </div>
      </main>
    </div>
  );
}

/** Replace raw source ID tags like [src_1, src_2] or [src_edgar] with (Source 1, Source 2). */
function formatContent(text: string): string {
  let counter = 0;
  return text.replace(/\[[^\]]*\bsrc[^\]]*\]/gi, (match) => {
    const parts = match
      .slice(1, -1)
      .split(",")
      .map((s) => s.trim())
      .filter((s) => /src/i.test(s));
    if (parts.length === 0) return match;
    const labels = parts.map(() => `Source ${++counter}`);
    return `(${labels.join(", ")})`;
  });
}

function EmptyState() {
  return (
    <div className="rounded-xl border border-dashed border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] backdrop-blur-sm p-12 text-center">
      <Sparkles className="mx-auto mb-3 h-8 w-8 text-[oklch(0.78_0.14_70)]" />
      <h3 className="text-lg font-semibold">Ready to analyse</h3>
      <p className="mt-1 text-sm text-white/60">
        Enter a company on the left and run the four specialist agents.
      </p>
    </div>
  );
}

function ProgressPanel({
  agents,
  synthesizing,
  done,
}: {
  agents: AgentRow[];
  synthesizing: boolean;
  done: boolean;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] p-5 backdrop-blur-sm">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold">
          🚀 {done ? "Pipeline complete" : "Running 5 specialist agents in parallel…"}
        </h3>
      </div>
      <ul className="space-y-2">
        {Object.keys(AGENT_META).map((key) => {
          const row = agents.find((a) => a.agent === key);
          const meta = AGENT_META[key];
          const status: AgentStatus = row?.status ?? "idle";
          return (
            <li
              key={key}
              className={cn(
                "flex items-center justify-between rounded-md border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] px-3 py-2.5 text-sm transition",
                status === "idle" && "opacity-50",
              )}
            >
              <div className="flex items-center gap-3">
                {status === "done" ? (
                  <CheckCircle2 className="h-4 w-4 text-success" />
                ) : status === "running" ? (
                  <Loader2 className="h-4 w-4 animate-spin text-[oklch(0.78_0.14_70)]" />
                ) : (
                  <span className="h-4 w-4 rounded-full border border-white/20" />
                )}
                <span>
                  <span className="mr-1">{meta.icon}</span>
                  {meta.label}
                </span>
              </div>
              {row?.status === "done" && (
                <div className="tnum flex items-center gap-3 text-xs text-white/60">
                  <span>{row.findings} findings</span>
                  <ConfidenceBadge value={row.confidence ?? 0} />
                </div>
              )}
            </li>
          );
        })}
        <li
          className={cn(
            "flex items-center justify-between rounded-md border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] px-3 py-2.5 text-sm",
            !synthesizing && !done && "opacity-50",
          )}
        >
          <div className="flex items-center gap-3">
            {done ? (
              <CheckCircle2 className="h-4 w-4 text-success" />
            ) : synthesizing ? (
              <Loader2 className="h-4 w-4 animate-spin text-[oklch(0.78_0.14_70)]" />
            ) : (
              <span className="h-4 w-4 rounded-full border border-white/20" />
            )}
            <span>🧠 Synthesis Agent — reconciling findings…</span>
          </div>
        </li>
      </ul>
    </div>
  );
}

const TABS = [
  { id: "summary", label: "📋 Executive Summary" },
  { id: "stats", label: "📊 Stats" },
  { id: "export", label: "⬇️ Export" },
] as const;

function MemoView({ memo }: { memo: InvestmentMemo }) {
  const [tab, setTab] = useState<string>("summary");
  const sectionTabs = memo.sections.map((s, i) => ({
    id: `s_${i}`,
    label: s.title,
  }));
  return (
    <div className="rounded-xl border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] backdrop-blur-sm">
      <div className="border-b border-white/10 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-wider text-white/70">
              Investment Memo
            </div>
            <h2 className="text-2xl font-bold text-foreground">
              {memo.company_name}
            </h2>
          </div>
          <ConfidenceBadge value={memo.overall_confidence} className="text-sm" />
        </div>
      </div>

      <div className="flex flex-wrap gap-1 border-b border-white/10 bg-[oklch(0.18_0.03_55_/_0.50)] px-3 pt-3">
        {[TABS[0], ...sectionTabs, TABS[1], TABS[2]].map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              "relative -mb-px rounded-t-md border border-transparent px-3 py-2 text-sm transition",
              tab === t.id
                ? "border-[oklch(0.75_0.14_65_/_0.60)] bg-[oklch(0.28_0.05_55_/_0.70)] text-white"
                : "text-white/55 hover:text-white",
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="p-6">
        {tab === "summary" && <SummaryTab memo={memo} />}
        {tab === "stats" && <StatsTab memo={memo} />}
        {tab === "export" && <ExportTab memo={memo} />}
        {sectionTabs.map(
          (st, i) =>
            tab === st.id && (
              <SectionTab
                key={st.id}
                section={memo.sections[i]}
                sourceRegistry={memo.metadata.source_registry ?? {}}
              />
            ),
        )}
      </div>
    </div>
  );
}

function SummaryTab({ memo }: { memo: InvestmentMemo }) {
  return (
    <div className="space-y-6">
      <p className="text-base leading-relaxed text-foreground/90">
        {formatContent(memo.executive_summary)}
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border border-success/30 bg-[oklch(0.20_0.03_55_/_0.75)] backdrop-blur-sm p-4">
          <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-success">
            <TrendingUp className="h-4 w-4" /> Investment Highlights
          </h4>
          <ul className="space-y-2 text-sm text-foreground/90">
            {memo.metadata.investment_highlights.map((h, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-success">✓</span>
                <span>{h}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="rounded-lg border border-warning/30 bg-[oklch(0.20_0.03_55_/_0.75)] backdrop-blur-sm p-4">
          <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-warning">
            <AlertTriangle className="h-4 w-4" /> Key Risks
          </h4>
          <ul className="space-y-2 text-sm text-foreground/90">
            {memo.metadata.investment_risks.map((r, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-warning">!</span>
                <span>{r}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function SectionTab({
  section,
  sourceRegistry,
}: {
  section: InvestmentMemo["sections"][number];
  sourceRegistry: NonNullable<InvestmentMemo["metadata"]["source_registry"]>;
}) {
  const [showClaims, setShowClaims] = useState(false);
  const [showConflicts, setShowConflicts] = useState(false);
  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-xl font-semibold">{section.title}</h3>
        <ConfidenceBadge value={section.confidence_score} />
      </div>
      <div className="prose prose-invert max-w-none whitespace-pre-wrap text-sm leading-relaxed text-foreground/90">
        {formatContent(section.content)}
      </div>
      <Collapsible
        open={showClaims}
        onOpenChange={setShowClaims}
        title={`📋 Supporting Claims (${section.claims.length})`}
      >
        <ul className="space-y-2">
          {section.claims.map((c, i) => (
            <li
              key={i}
              className="rounded-md border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] p-3 text-sm"
            >
              <div className="flex items-start justify-between gap-3">
                <span className="text-foreground/90">{formatContent(c.text)}</span>
                <div className="flex shrink-0 items-center gap-2">
                  <ConfidenceBadge value={c.confidence} />
                  <span className="tnum rounded-full border border-white/10 bg-[oklch(0.18_0.02_55_/_0.60)] px-2 py-0.5 text-xs text-white/60">
                    {c.source_ids.length} {c.source_ids.length === 1 ? "source" : "sources"}
                  </span>
                </div>
              </div>
              {c.source_ids.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {c.source_ids.map((sid, sourceIndex) => {
                    const src = sourceRegistry[sid];
                    if (!src?.url) {
                      return (
                        <span
                          key={sid}
                          className="rounded-full border border-white/10 px-2 py-0.5 text-xs text-white/50"
                        >
                          Source {sourceIndex + 1}
                        </span>
                      );
                    }
                    return (
                      <a
                        key={sid}
                        href={src.url}
                        target="_blank"
                        rel="noreferrer"
                        title={`${src.title}${src.originating_agent ? ` • ${src.originating_agent}` : ""}`}
                        className="rounded-full border border-white/15 bg-white/5 px-2 py-0.5 text-xs text-white/65 transition hover:border-[oklch(0.75_0.14_65)] hover:text-white"
                      >
                        Source {sourceIndex + 1}
                        {src.originating_agent ? ` · ${src.originating_agent.replace("_", " ")}` : ""}
                      </a>
                    );
                  })}
                </div>
              )}
            </li>
          ))}
        </ul>
      </Collapsible>
      {section.conflicting_claims.length > 0 && (
        <Collapsible
          open={showConflicts}
          onOpenChange={setShowConflicts}
          title={`⚠️ Cross-Agent Conflicts (${section.conflicting_claims.length})`}
        >
          <ul className="space-y-3">
            {section.conflicting_claims.map((c, i) => (
              <li
                key={i}
                className="rounded-md border border-warning/40 bg-[oklch(0.20_0.03_55_/_0.75)] backdrop-blur-sm p-4 text-sm"
              >
                <div className="mb-3 font-medium text-warning">{c.description}</div>
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="rounded border border-white/10 bg-[oklch(0.18_0.02_55_/_0.75)] p-3">
                    <div className="mb-1 text-xs font-semibold uppercase text-white/60">
                      Claim A
                    </div>
                    <div className="text-foreground/90">{c.claim_a.text}</div>
                    <ConfidenceBadge value={c.claim_a.confidence} className="mt-2" />
                  </div>
                  <div className="rounded border border-white/10 bg-[oklch(0.18_0.02_55_/_0.75)] p-3">
                    <div className="mb-1 text-xs font-semibold uppercase text-white/60">
                      Claim B
                    </div>
                    <div className="text-foreground/90">{c.claim_b.text}</div>
                    <ConfidenceBadge value={c.claim_b.confidence} className="mt-2" />
                  </div>
                </div>
                {c.resolution && (
                  <div className="mt-3 text-xs text-white/60">
                    <span className="font-semibold">Resolution:</span> {c.resolution}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </Collapsible>
      )}
    </div>
  );
}

function StatsTab({ memo }: { memo: InvestmentMemo }) {
  const m = memo.metadata;
  const allClaims = memo.sections.flatMap((s) => s.claims);
  const sourced = allClaims.filter((c) => c.source_ids.length > 0).length;
  const faithfulness = allClaims.length ? sourced / allClaims.length : 0;
  const verification = m.verification;
  const flaggedClaims = verification
    ? verification.weak_claims +
      verification.unsupported_claims +
      verification.missing_source_claims +
      verification.unresolved_source_claims
    : 0;
  const cards = [
    { label: "Findings", value: m.total_findings },
    { label: "Sources", value: m.total_sources },
    { label: "Agents", value: m.agent_count },
    { label: "Run time", value: `${m.elapsed_seconds.toFixed(1)}s` },
  ];
  const confEntries = Object.entries(m.specialist_confidences);
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {cards.map((c) => (
          <div
            key={c.label}
            className="rounded-lg border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] p-4"
          >
            <div className="text-xs uppercase tracking-wider text-white/70">
              {c.label}
            </div>
            <div className="tnum mt-1 text-2xl font-bold">{c.value}</div>
          </div>
        ))}
      </div>
      <div className="rounded-lg border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] p-5">
        <h4 className="mb-3 text-sm font-semibold">Agent confidence</h4>
        <div className="space-y-3">
          {confEntries.map(([agent, conf]) => {
            const meta = AGENT_META[agent];
            const tone =
              conf >= 0.7 ? "bg-success" : conf >= 0.5 ? "bg-warning" : "bg-danger";
            return (
              <div key={agent}>
                <div className="mb-1 flex items-center justify-between text-xs">
                  <span>
                    {meta?.icon} {meta?.label ?? agent}
                  </span>
                  <span className="tnum text-white/60">
                    {Math.round(conf * 100)}%
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-[oklch(0.15_0.02_55_/_0.80)]">
                  <div
                    className={cn("h-full rounded-full", tone)}
                    style={{ width: `${Math.round(conf * 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-lg border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] p-5">
          <div className="text-xs uppercase tracking-wider text-white/70">
            Faithfulness
          </div>
          <div className="tnum mt-1 text-3xl font-bold">
            {Math.round(faithfulness * 100)}%
          </div>
          <div className="text-xs text-white/60">
            {sourced} of {allClaims.length} claims have sources
          </div>
        </div>
        <div className="rounded-lg border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] p-5">
          <div className="text-xs uppercase tracking-wider text-white/70">
            Verification
          </div>
          <div className="tnum mt-1 text-3xl font-bold">
            {verification ? `${Math.round(verification.overall_score * 100)}%` : "—"}
          </div>
          <div className="text-xs text-white/60">
            {verification
              ? `${verification.supported_claims}/${verification.total_claims} supported · ${flaggedClaims} need review`
              : "Run a fresh memo to score source support"}
          </div>
        </div>
      </div>
      {verification && flaggedClaims > 0 && (
        <div className="rounded-lg border border-white/10 bg-[oklch(0.20_0.03_55_/_0.65)] p-5">
          <h4 className="mb-3 text-sm font-semibold">Claims needing review</h4>
          <div className="space-y-2">
            {verification.per_claim
              .filter((c) => c.status !== "supported")
              .slice(0, 8)
              .map((claim, index) => (
                <div
                  key={`${claim.claim_text}-${index}`}
                  className="rounded-md border border-white/10 bg-[oklch(0.18_0.02_55_/_0.75)] p-3 text-sm"
                >
                  <div className="flex flex-wrap items-center gap-2 text-xs text-white/60">
                    <span className="rounded border border-white/10 px-2 py-0.5 uppercase">
                      {claim.status.replace("_", " ")}
                    </span>
                    <span>{claim.section_title}</span>
                  </div>
                  <p className="mt-2 text-white/80">{claim.claim_text}</p>
                  <p className="mt-1 text-xs text-white/50">{claim.reason}</p>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ExportTab({ memo }: { memo: InvestmentMemo }) {
  function download() {
    const blob = new Blob([JSON.stringify(memo, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${memo.company_name.replace(/\s+/g, "_")}_memo.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <div className="space-y-4">
      <p className="text-sm text-white/60">
        Download the full investment memo as JSON.
      </p>
      <button
        onClick={download}
        className="inline-flex items-center gap-2 rounded-md bg-[oklch(0.55_0.15_55)] px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-black/20 hover:bg-[oklch(0.62_0.17_58)]"
      >
        <Download className="h-4 w-4" /> Download memo.json
      </button>
    </div>
  );
}

function Collapsible({
  open,
  onOpenChange,
  title,
  children,
}: {
  open: boolean;
  onOpenChange: (b: boolean) => void;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-[oklch(0.22_0.04_55_/_0.55)] backdrop-blur-sm">
      <button
        onClick={() => onOpenChange(!open)}
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium"
      >
        <span>{title}</span>
        <ChevronDown
          className={cn("h-4 w-4 transition", open && "rotate-180")}
        />
      </button>
      {open && <div className="border-t border-white/10 p-4">{children}</div>}
    </div>
  );
}
