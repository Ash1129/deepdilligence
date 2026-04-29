"""Recommendation engine — reads all memos and produces ranked BUY/HOLD/SELL ratings.

Acts like a portfolio manager / IB analyst sitting above the four specialist agents.
It reads every company's memo, compares them side-by-side, and produces a ranked
weekly investment recommendation report.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from src.models.schemas import InvestmentMemo
from src.utils.config import SYNTHESIS_MODEL, get_openai_api_key

logger = logging.getLogger(__name__)

RECOMMENDATIONS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "recommendations"
)

# ─── Output schemas ───────────────────────────────────────────────────────────

RATING_OPTIONS = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]


class StockRating(BaseModel):
    """Single company rating produced by the recommendation engine."""
    company_name: str
    ticker: str
    sector: str
    rating: str = Field(..., description="One of: STRONG BUY, BUY, HOLD, SELL, STRONG SELL")
    rank: int = Field(..., description="1 = best pick, N = worst pick across the universe")
    bull_case: str = Field(..., description="Top 1-2 sentence bull case")
    bear_case: str = Field(..., description="Top 1-2 sentence bear case")
    rationale: str = Field(..., description="2-3 sentence rating rationale")
    suggested_weight_pct: float = Field(..., description="Suggested portfolio weight 0-100")
    confidence: float = Field(..., ge=0.0, le=1.0)


class WeeklyReport(BaseModel):
    """Full weekly recommendation report across all analysed companies."""
    model_config = {"protected_namespaces": ()}

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    week_of: str = Field(..., description="ISO week string e.g. '2026-W18'")
    universe_size: int
    ratings: list[StockRating]       # sorted best → worst
    top_picks: list[str]             # tickers of top 3
    avoid: list[str]                 # tickers rated SELL or STRONG SELL
    macro_commentary: str            # 2-3 sentence market view
    sector_views: dict[str, str]     # sector → "Bullish" | "Neutral" | "Bearish"
    model_used: str


# ─── OpenAI tool definition ───────────────────────────────────────────────────

_RECOMMEND_TOOL = {
    "type": "function",
    "function": {
        "name": "produce_weekly_recommendations",
        "description": (
            "Produce a ranked weekly investment recommendation report across all analysed companies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ratings": {
                    "type": "array",
                    "description": "All companies ranked best to worst",
                    "items": {
                        "type": "object",
                        "properties": {
                            "company_name": {"type": "string"},
                            "ticker":       {"type": "string"},
                            "sector":       {"type": "string"},
                            "rating":       {"type": "string", "enum": RATING_OPTIONS},
                            "rank":         {"type": "integer"},
                            "bull_case":    {"type": "string"},
                            "bear_case":    {"type": "string"},
                            "rationale":    {"type": "string"},
                            "suggested_weight_pct": {"type": "number"},
                            "confidence":   {"type": "number"},
                        },
                        "required": ["company_name","ticker","sector","rating",
                                     "rank","bull_case","bear_case","rationale",
                                     "suggested_weight_pct","confidence"],
                    },
                },
                "top_picks":         {"type": "array", "items": {"type": "string"}},
                "avoid":             {"type": "array", "items": {"type": "string"}},
                "macro_commentary":  {"type": "string"},
                "sector_views": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["ratings","top_picks","avoid","macro_commentary","sector_views"],
        },
    },
}

_SYSTEM_PROMPT = """\
You are a senior portfolio manager at a top-tier investment bank, acting as the
final decision-maker above a team of specialist analysts (financial, team/culture,
market, and risk/sentiment).

You will receive a set of due-diligence memos, one per company.  Your job is to:

1. Read each memo carefully — executive summary, section findings, confidence
   scores, cross-agent conflicts, and highlighted risks/strengths.
2. Compare companies against each other on a consistent framework:
   - Financial health and growth trajectory
   - Competitive moat and market positioning
   - Team quality and execution track record
   - Risk-adjusted return potential
3. Assign a rating to every company: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
4. Rank them 1 (best) to N (worst).
5. Suggest a portfolio weight for each BUY/STRONG BUY (weights must sum to ≤100%).
6. Provide a concise macro commentary and sector views.

Be decisive. Differentiate clearly between companies — do not rate everything HOLD.
Every claim in your rationale must be traceable to specific findings in the memos.

IMPORTANT — keep your output SHORT to avoid truncation:
- bull_case: 1 sentence max
- bear_case: 1 sentence max
- rationale:  2 sentences max
- macro_commentary: 2 sentences max
"""


# ─── Recommender ─────────────────────────────────────────────────────────────

class RecommendationEngine:
    """Reads all memos and produces a ranked WeeklyReport via LLM."""

    def __init__(self) -> None:
        self._client = OpenAI(api_key=get_openai_api_key())

    def generate(self, memos: dict[str, InvestmentMemo]) -> WeeklyReport:
        """Generate a weekly recommendation report from a dict of ticker → memo.

        Args:
            memos: Output of BatchRunner.run_batch() — ticker → InvestmentMemo.

        Returns:
            WeeklyReport with ranked ratings for every company.
        """
        if not memos:
            raise ValueError("No memos provided — run batch first")

        logger.info("Generating recommendations for %d companies", len(memos))

        # Format all memos into a single prompt
        memo_text = self._format_memos(memos)

        user_message = (
            f"You are evaluating {len(memos)} companies for this week's investment report.\n\n"
            f"{memo_text}\n\n"
            "Produce your ranked recommendation report now. "
            "Call produce_weekly_recommendations with your complete analysis."
        )

        response = self._client.chat.completions.create(
            model=SYNTHESIS_MODEL,
            messages=[
                {"role": "system",  "content": _SYSTEM_PROMPT},
                {"role": "user",    "content": user_message},
            ],
            tools=[_RECOMMEND_TOOL],
            tool_choice={"type": "function", "function": {"name": "produce_weekly_recommendations"}},
            max_completion_tokens=32000,
        )

        message = response.choices[0].message
        if not message.tool_calls:
            raise RuntimeError("Recommendation tool not called by model")

        raw_args = message.tool_calls[0].function.arguments
        try:
            raw = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            # Response was truncated — try to salvage by truncating to last complete object
            logger.warning("Tool call JSON truncated (%s) — attempting partial recovery", exc)
            raw = self._recover_truncated_json(raw_args)

        return self._build_report(raw, memos)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _format_memos(self, memos: dict[str, InvestmentMemo]) -> str:
        """Render all memos as compact structured text for the recommendation prompt.

        Per-company budget is intentionally tight so many companies can fit in
        one context without truncating the model's output JSON.
        """
        # Allocate fewer chars per company when the universe is large
        n = len(memos)
        summary_chars  = max(200, 600  // max(n // 5, 1))
        section_chars  = max(100, 250  // max(n // 5, 1))

        parts: list[str] = []
        for ticker, memo in memos.items():
            parts.append(f"\n{'='*50}")
            parts.append(f"{memo.company_name} ({ticker}) | conf={memo.overall_confidence:.0%}")

            parts.append(memo.executive_summary[:summary_chars])

            hi = memo.metadata.get("investment_highlights", [])
            ri = memo.metadata.get("investment_risks", [])
            if hi:
                parts.append("STRENGTHS: " + " | ".join(hi[:2]))
            if ri:
                parts.append("RISKS: " + " | ".join(ri[:2]))

            for section in memo.sections:
                n_conflicts = len(section.conflicting_claims)
                conflict_note = f" ⚠{n_conflicts}conflicts" if n_conflicts else ""
                parts.append(
                    f"[{section.title} {section.confidence_score:.0%}{conflict_note}] "
                    f"{section.content[:section_chars]}"
                )

        return "\n".join(parts)

    @staticmethod
    def _recover_truncated_json(raw: str) -> dict:
        """Best-effort recovery when the model's tool-call JSON is truncated.

        Uses regex to find all complete rating objects (each ends with a
        "confidence": <number> } pattern), then wraps them in a minimal valid
        top-level structure with safe defaults for missing fields.
        """
        import re

        # Every complete rating object ends with a confidence value field.
        # Find where the last one ends.
        confidence_hits = list(re.finditer(r'"confidence"\s*:\s*[\d.]+\s*\}', raw))
        if not confidence_hits:
            raise RuntimeError(
                "Recommendation JSON is truncated and could not be recovered. "
                "Try again with fewer companies."
            )

        last_rating_end = confidence_hits[-1].end()

        # Find where the ratings array opens
        ratings_key = raw.find('"ratings"')
        if ratings_key == -1:
            raise RuntimeError(
                "Recommendation JSON is truncated and could not be recovered. "
                "Try again with fewer companies."
            )
        array_open = raw.find("[", ratings_key)
        if array_open == -1:
            raise RuntimeError(
                "Recommendation JSON is truncated and could not be recovered. "
                "Try again with fewer companies."
            )

        # Slice out just the ratings array content (possibly incomplete)
        ratings_fragment = raw[array_open : last_rating_end]

        # Build a minimal valid top-level object with defaults for fields
        # that were not yet emitted when the response was cut off.
        candidate = (
            '{"ratings": ' + ratings_fragment + '], '
            '"top_picks": [], '
            '"avoid": [], '
            '"macro_commentary": "Report truncated — rerun with fewer companies for full analysis.", '
            '"sector_views": {"General": "See individual ratings above"}}'
        )

        try:
            result = json.loads(candidate)
            logger.info(
                "Partial JSON recovery succeeded — %d ratings recovered",
                len(result.get("ratings", [])),
            )
            return result
        except json.JSONDecodeError:
            raise RuntimeError(
                "Recommendation JSON is truncated and could not be recovered. "
                "Try again with fewer companies."
            )

    def _build_report(self, raw: dict, memos: dict[str, InvestmentMemo]) -> WeeklyReport:
        """Convert raw tool output into a validated WeeklyReport."""
        now = datetime.utcnow()
        week_of = f"{now.year}-W{now.isocalendar()[1]:02d}"

        ratings: list[StockRating] = []
        for r in raw.get("ratings", []):
            ratings.append(StockRating(
                company_name=str(r.get("company_name", "")),
                ticker=str(r.get("ticker", "")),
                sector=str(r.get("sector", "")),
                rating=str(r.get("rating", "HOLD")),
                rank=int(r.get("rank", 99)),
                bull_case=str(r.get("bull_case", "")),
                bear_case=str(r.get("bear_case", "")),
                rationale=str(r.get("rationale", "")),
                suggested_weight_pct=float(r.get("suggested_weight_pct", 0.0)),
                confidence=float(max(0.0, min(1.0, r.get("confidence", 0.5)))),
            ))

        # Sort by rank ascending (rank 1 = best)
        ratings.sort(key=lambda x: x.rank)

        return WeeklyReport(
            generated_at=now,
            week_of=week_of,
            universe_size=len(memos),
            ratings=ratings,
            top_picks=raw.get("top_picks", [])[:3],
            avoid=raw.get("avoid", []),
            macro_commentary=str(raw.get("macro_commentary", "")),
            sector_views=raw.get("sector_views", {}),
            model_used=SYNTHESIS_MODEL,
        )


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_report(report: WeeklyReport) -> Path:
    """Save a WeeklyReport to data/recommendations/."""
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"report_{report.week_of.replace('-', '_')}.json"
    path = RECOMMENDATIONS_DIR / fname
    with open(path, "w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2)
    logger.info("Weekly report saved → %s", path)
    return path


def load_latest_report() -> WeeklyReport | None:
    """Load the most recent weekly report from disk, or None if none exist."""
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RECOMMENDATIONS_DIR.glob("report_*.json"), reverse=True)
    if not files:
        return None
    try:
        with open(files[0]) as f:
            return WeeklyReport.model_validate(json.load(f))
    except Exception as e:
        logger.warning("Failed to load latest report: %s", e)
        return None


def load_all_reports() -> list[WeeklyReport]:
    """Load all historical weekly reports, newest first."""
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[WeeklyReport] = []
    for path in sorted(RECOMMENDATIONS_DIR.glob("report_*.json"), reverse=True):
        try:
            with open(path) as f:
                reports.append(WeeklyReport.model_validate(json.load(f)))
        except Exception as e:
            logger.warning("Skipping %s: %s", path.name, e)
    return reports
