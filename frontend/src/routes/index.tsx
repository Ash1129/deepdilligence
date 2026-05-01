import { createFileRoute } from "@tanstack/react-router";
import { Link, useNavigate } from "@tanstack/react-router";
import { Header } from "@/components/Header";
import { ArrowRight } from "lucide-react";
import heroSkyscraper from "@/assets/hero-skyscraper.jpg";

export const Route = createFileRoute("/")({
  component: Index,
});

const QUICK_PICKS = [
  { company: "Apple Inc", ticker: "AAPL" },
  { company: "Microsoft", ticker: "MSFT" },
  { company: "NVIDIA", ticker: "NVDA" },
  { company: "Tesla", ticker: "TSLA" },
  { company: "Amazon", ticker: "AMZN" },
];


function Index() {
  const navigate = useNavigate();
  return (
    <div className="relative min-h-screen">
      {/* Hero background image — fades into page background */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 -z-0 h-[110vh] overflow-hidden"
      >
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${heroSkyscraper})` }}
        />
        {/* Tint with warm amber sampled from the image, then fade into page bg */}
        <div className="absolute inset-0 bg-[linear-gradient(180deg,_oklch(0.25_0.05_60_/_0.15)_0%,_oklch(0.22_0.06_55_/_0.35)_45%,_oklch(0.20_0.05_50_/_0.85)_75%,_var(--background)_100%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_oklch(0.78_0.14_70_/_0.18),_transparent_60%)]" />
      </div>
      <Header />
      <main className="relative z-10 mx-auto max-w-7xl px-6 pb-24">
        {/* Hero */}
        <section className="relative pt-36 pb-16 text-center">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/40 px-3 py-1 text-xs text-foreground/80 backdrop-blur">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-success" />
            Multi-agent equity research
          </div>
          <h1 className="text-balance text-5xl font-bold tracking-tight text-white drop-shadow-[0_2px_18px_rgba(0,0,0,0.65)] sm:text-6xl">
            AI-Powered Investment
            <br /> Due Diligence
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-white/85 drop-shadow-[0_1px_10px_rgba(0,0,0,0.6)]">
            Four specialist agents. One investment memo. Weekly rankings.
          </p>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              to="/portfolio"
              className="group inline-flex items-center gap-2 rounded-md bg-[oklch(0.55_0.15_55)] px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-black/20 transition hover:bg-[oklch(0.62_0.17_58)]"
            >
              Portfolio Builder
              <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
            </Link>
            <Link
              to="/weekly"
              className="inline-flex items-center gap-2 rounded-md border border-white/15 bg-black/40 px-5 py-2.5 text-sm font-semibold text-foreground backdrop-blur transition hover:bg-black/60"
            >
              Weekly Rankings
              <ArrowRight className="h-4 w-4" />
            </Link>
            <button
              onClick={() => navigate({ to: "/analyze" })}
              className="inline-flex items-center gap-2 rounded-md border border-white/15 bg-black/40 px-5 py-2.5 text-sm font-semibold text-foreground backdrop-blur transition hover:bg-black/60"
            >
              Analyse a Company
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>

          {/* Quick picks */}
          <div className="mt-10">
            <p className="mb-3 text-xs uppercase tracking-wider text-white/70">
              Quick start
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              {QUICK_PICKS.map((q) => (
                <button
                  key={q.ticker}
                  onClick={() =>
                    navigate({
                      to: "/analyze",
                      search: { company: q.company, ticker: q.ticker },
                    })
                  }
                  className="tnum rounded-md border border-white/10 bg-black/40 px-3 py-1.5 text-sm text-foreground backdrop-blur transition hover:border-primary/60 hover:bg-black/60"
                >
                  {q.company}{" "}
                  <span className="text-white/60">({q.ticker})</span>
                </button>
              ))}
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}
