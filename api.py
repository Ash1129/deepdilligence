"""DeepDiligence FastAPI backend.

Exposes the full pipeline as a REST + Server-Sent Events API so any frontend
(React, Next.js, etc.) can connect without going through Streamlit.

Run with:
    uvicorn api:app --reload --port 8000

Endpoints
---------
GET  /api/health
GET  /api/weekly-report            — latest saved report
GET  /api/weekly-report/all        — all historical reports
GET  /api/universe                 — full curated stock universe
POST /api/analyze                  — SSE stream: single-company pipeline
POST /api/weekly-report/generate   — SSE stream: batch pipeline + recommendations
"""

from __future__ import annotations

import json
import logging
import queue
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.models.schemas import AgentSubReport
from src.orchestrator import Orchestrator
from src.scheduler.portfolio_builder import build_portfolio
from src.scheduler.batch_runner import _run_one
from src.scheduler.recommender import (
    RecommendationEngine,
    load_all_reports,
    load_latest_report,
    save_report,
)
from src.scheduler.screener import ALL_SECTORS, ALL_STYLES, build_watchlist, load_universe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="DeepDiligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your Lovable/Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request models ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    company: str
    ticker: str | None = None


class WeeklyReportRequest(BaseModel):
    styles: list[str] = []
    sectors: list[str] = []
    top_n: int = 10
    use_screener: bool = False
    screener_criteria: str = "Price Change"


class PortfolioRequest(BaseModel):
    amount: float
    max_positions: int = 10
    max_position_pct: float = 30.0
    strong_buy_only: bool = False


# ─── SSE helper ───────────────────────────────────────────────────────────────

def _sse(event_type: str, data: dict) -> str:
    """Encode one Server-Sent Event message."""
    payload = json.dumps({"type": event_type, **data})
    return f"data: {payload}\n\n"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "DeepDiligence"}


@app.get("/api/universe")
def get_universe() -> dict:
    """Return the full curated stock universe with style and sector metadata."""
    return {
        "stocks": load_universe(),
        "all_styles": ALL_STYLES,
        "all_sectors": ALL_SECTORS,
    }


@app.get("/api/weekly-report")
def get_latest_weekly_report() -> dict:
    """Return the most recent saved weekly report."""
    report = load_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No weekly report found yet")
    return report.model_dump(mode="json")


@app.get("/api/weekly-report/all")
def get_all_weekly_reports() -> list:
    """Return all historical weekly reports, newest first."""
    return [r.model_dump(mode="json") for r in load_all_reports()]


@app.post("/api/analyze")
def analyze_company(req: AnalyzeRequest) -> StreamingResponse:
    """Run the full 4-agent + synthesis pipeline for one company.

    Returns a Server-Sent Events stream so the frontend can show live progress.

    Event types emitted:
        agent_start      { agent: str }
        agent_complete   { agent: str, findings: int, confidence: float }
        synthesis_start  {}
        complete         { memo: InvestmentMemo (JSON) }
        error            { message: str }
    """
    q: queue.Queue[str | None] = queue.Queue()

    def on_agent_start(name: str) -> None:
        # Fires from a worker thread — only queue-safe ops here
        q.put(_sse("agent_start", {"agent": name}))

    def on_agent_complete(name: str, report: AgentSubReport) -> None:
        q.put(_sse("agent_complete", {
            "agent":      name,
            "findings":   len(report.findings),
            "confidence": round(report.confidence_score, 4),
        }))

    def on_synthesis_start() -> None:
        q.put(_sse("synthesis_start", {}))

    def run() -> None:
        try:
            orch = Orchestrator(
                company_name=req.company,
                ticker=req.ticker or None,
                on_agent_start=on_agent_start,
                on_agent_complete=on_agent_complete,
                on_synthesis_start=on_synthesis_start,
            )
            memo = orch.run()
            q.put(_sse("complete", {"memo": json.loads(memo.model_dump_json())}))
        except Exception as exc:
            logger.exception("Pipeline failed for %s", req.company)
            q.put(_sse("error", {"message": str(exc)}))
        finally:
            q.put(None)  # signals end of stream

    threading.Thread(target=run, daemon=True).start()

    def event_stream():
        while True:
            msg = q.get()
            if msg is None:
                break
            yield msg

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/portfolio/build")
def build_portfolio_endpoint(req: PortfolioRequest) -> dict:
    """Build a portfolio allocation from the latest weekly report.

    Uses BUY / STRONG BUY rated stocks, weighs by confidence and suggested
    allocation, fetches live prices, and returns holdings with S&P 500 benchmark.
    """
    if req.amount <= 0:
        raise HTTPException(status_code=400, detail="Investment amount must be positive.")
    if not (1 <= req.max_positions <= 20):
        raise HTTPException(status_code=400, detail="max_positions must be between 1 and 20.")

    result = build_portfolio(
        amount=req.amount,
        max_positions=req.max_positions,
        max_position_pct=req.max_position_pct,
        strong_buy_only=req.strong_buy_only,
    )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["message"])

    return result


@app.post("/api/weekly-report/generate")
def generate_weekly_report(req: WeeklyReportRequest) -> StreamingResponse:
    """Build watchlist, run batch pipeline, generate recommendations.

    Returns a Server-Sent Events stream.

    Event types emitted:
        watchlist_start    {}
        watchlist_complete { stocks: [{ ticker, company, sector }] }
        stock_start        { ticker: str, company: str }
        stock_complete     { ticker: str, sections: int, confidence: float }
        stock_error        { ticker: str, message: str }
        ranking_start      { count: int }
        complete           { report: WeeklyReport (JSON) }
        error              { message: str }
    """
    q: queue.Queue[str | None] = queue.Queue()

    def run() -> None:
        try:
            # Step 1 — build watchlist
            q.put(_sse("watchlist_start", {}))
            selected = build_watchlist(
                use_screener=req.use_screener,
                screener_criteria=req.screener_criteria,
                styles=req.styles,
                sectors=req.sectors,
                top_n=req.top_n,
            )
            q.put(_sse("watchlist_complete", {
                "stocks": [
                    {"ticker": s["ticker"], "company": s["company"], "sector": s["sector"]}
                    for s in selected
                ],
            }))

            # Step 2 — run pipeline per stock
            memos: dict = {}
            for stock in selected:
                q.put(_sse("stock_start", {
                    "ticker":  stock["ticker"],
                    "company": stock["company"],
                }))
                try:
                    _, memo = _run_one(stock, force_refresh=False)
                    if memo:
                        memos[stock["ticker"]] = memo
                        q.put(_sse("stock_complete", {
                            "ticker":     stock["ticker"],
                            "sections":   len(memo.sections),
                            "confidence": round(memo.overall_confidence, 4),
                        }))
                    else:
                        q.put(_sse("stock_error", {
                            "ticker":  stock["ticker"],
                            "message": "pipeline returned no memo",
                        }))
                except Exception as exc:
                    q.put(_sse("stock_error", {
                        "ticker":  stock["ticker"],
                        "message": str(exc),
                    }))

            if not memos:
                q.put(_sse("error", {"message": "All pipelines failed — no memos to rank"}))
                return

            # Step 3 — generate recommendations
            q.put(_sse("ranking_start", {"count": len(memos)}))
            engine = RecommendationEngine()
            report = engine.generate(memos)
            save_report(report)
            q.put(_sse("complete", {"report": json.loads(report.model_dump_json())}))

        except Exception as exc:
            logger.exception("Weekly report generation failed")
            q.put(_sse("error", {"message": str(exc)}))
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    def event_stream():
        while True:
            msg = q.get()
            if msg is None:
                break
            yield msg

    return StreamingResponse(event_stream(), media_type="text/event-stream")
