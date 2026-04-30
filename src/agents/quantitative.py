"""Quantitative Momentum Agent — ML-based directional prediction from price history.

Unlike the four qualitative specialist agents, this agent does NOT use the ReAct
tool-calling loop for data gathering.  Its pipeline is:

    gather_data()
        └── yfinance download  →  feature engineering  →  RandomForest prediction
            (fully deterministic — no LLM involved at this stage)

    analyze(raw_data)
        └── LLM interprets ML output → structured AgentSubReport with claims,
            confidence calibrated to holdout accuracy, and epistemic caveats.

For private companies or tickers with < 120 trading days of history, the agent
returns a graceful zero-confidence stub (same pattern as other agents on failure).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.agents.base_agent import BaseAgent
from src.data.price_history import fetch_price_history, run_ml_prediction
from src.models.prompts import PRODUCE_ANALYSIS_TOOL, QUANT_ANALYZE_SYSTEM
from src.models.schemas import AgentClaim, AgentSubReport, Source, SourceType
from src.utils.config import AGENT_MODEL, CACHE_DIR, get_openai_api_key

logger = logging.getLogger(__name__)


class QuantitativeAgent(BaseAgent):
    """Random-Forest momentum signal agent.

    Trains a RandomForestClassifier on 2-3 years of daily OHLCV data and
    predicts the stock's price direction over the next 20 trading days
    (UP >+3 %, FLAT ±3 %, DOWN <-3 %).

    Key design decisions:
    - Extends BaseAgent directly (not ReactAgent) — data gathering is
      deterministic, not LLM-driven.
    - LLM is only used in analyze() to translate ML numbers into natural-
      language claims with appropriate caveats.
    - Inherits disk-cache from ReactAgent's run() override — not applicable
      here, so we re-implement a lightweight cache via BaseAgent.run().
    - All claims point to a single source: the Yahoo Finance price history page.
    """

    def __init__(self, company_name: str, ticker: str | None = None, **kwargs: Any) -> None:
        super().__init__(company_name, ticker, **kwargs)
        self._client = OpenAI(api_key=get_openai_api_key())

    # ─── Identity ────────────────────────────────────────────────────────────

    @property
    def agent_name(self) -> str:
        return "quantitative_momentum"

    @property
    def description(self) -> str:
        return (
            "Applies a Random Forest classifier to 2-3 years of daily price/volume data "
            "to produce a 20-trading-day directional signal (UP / FLAT / DOWN) with "
            "probability estimates, holdout accuracy, and a technical regime snapshot. "
            "Applicable only to publicly traded companies with sufficient price history."
        )

    # ─── Data gathering (deterministic) ──────────────────────────────────────

    # ─── Disk cache (mirrors ReactAgent pattern) ─────────────────────────────

    def _cache_path(self) -> Path:
        raw = f"{self.agent_name}|{self.company_name}|{self.ticker or ''}"
        key = hashlib.sha256(raw.encode()).hexdigest()[:16]
        cache_dir = CACHE_DIR / "agents" / self.agent_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key}_report.json"

    def run(self) -> AgentSubReport:
        """Run with disk caching — same pattern as ReactAgent.run()."""
        path = self._cache_path()
        if path.exists():
            self.logger.info("Cache hit — loading quantitative report from %s", path.name)
            with open(path) as f:
                return AgentSubReport.model_validate(json.load(f))

        report = super().run()

        with open(path, "w") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2)
        self.logger.info("Cached quantitative report → %s", path.name)
        return report

    # ─── Data gathering (deterministic) ──────────────────────────────────────

    def gather_data(self) -> dict[str, Any]:
        """Download price history and run the ML prediction pipeline.

        Returns a dict containing:
            - 'ticker'    (str)
            - 'ml_result' (dict from run_ml_prediction)

        Or an error dict if the ticker is missing or data is insufficient:
            - 'error'   (str: error code)
            - 'message' (str: human-readable explanation)
        """
        if not self.ticker:
            return {
                "error": "no_ticker",
                "message": (
                    f"No ticker available for {self.company_name}. "
                    "Quantitative analysis requires a publicly traded stock ticker."
                ),
            }

        # Retry up to 3 times — yfinance can fail transiently on first call
        df = None
        for attempt in range(3):
            self.logger.info(
                "Fetching price history for %s (attempt %d/3)", self.ticker, attempt + 1
            )
            df = fetch_price_history(self.ticker, years=3)
            if df is not None:
                break
            if attempt < 2:
                import time
                time.sleep(1.5)

        if df is None:
            return {
                "error": "insufficient_data",
                "message": (
                    f"Insufficient price history retrieved for {self.ticker} after 3 attempts. "
                    "The ticker may be unlisted, too recently listed, or delisted."
                ),
            }

        self.logger.info(
            "Running ML prediction for %s (%d rows)", self.ticker, len(df)
        )
        ml_result = run_ml_prediction(df)

        if "error" in ml_result:
            return {
                "error": ml_result["error"],
                "message": (
                    f"ML pipeline failed for {self.ticker}: {ml_result['error']}. "
                    "Insufficient clean feature rows after indicator calculation."
                ),
            }

        return {
            "ticker":    self.ticker,
            "ml_result": ml_result,
        }

    # ─── Analysis (LLM interprets ML output) ─────────────────────────────────

    def analyze(self, raw_data: dict[str, Any]) -> AgentSubReport:
        """Translate the ML prediction into a structured AgentSubReport.

        If gather_data() returned an error, returns a graceful zero-confidence stub.
        Otherwise, passes the ML output to the LLM with QUANT_ANALYZE_SYSTEM prompt
        and forces a structured produce_analysis tool call.

        Args:
            raw_data: Output of gather_data().

        Returns:
            Validated AgentSubReport.
        """
        # ── Handle upstream errors gracefully ──
        if "error" in raw_data:
            return self._error_stub(raw_data.get("message", raw_data["error"]))

        ticker    = raw_data["ticker"]
        ml_result = raw_data["ml_result"]

        if "error" in ml_result:
            return self._error_stub(
                f"ML pipeline error ({ml_result['error']}) for {ticker}."
            )

        # ── Build a single, traceable source for all claims ──
        source = Source(
            id="src_price_history",
            url=f"https://finance.yahoo.com/quote/{ticker}/history/",
            title=(
                f"{self.company_name} ({ticker}) — Daily Price History "
                f"({ml_result['data_start']} to {ml_result['data_end']}) via Yahoo Finance"
            ),
            retrieved_at=datetime.utcnow(),
            snippet=(
                f"Daily OHLCV data: {ml_result['training_samples']} training samples, "
                f"{ml_result['holdout_samples']} holdout samples. "
                f"RandomForest holdout accuracy: {ml_result['holdout_accuracy']:.2%}."
            ),
            source_type=SourceType.OTHER,
        )

        # ── Ask the LLM to produce structured findings ──
        user_message = (
            f"Company: {self.company_name}  |  Ticker: {ticker}\n\n"
            "=== RANDOM FOREST ML OUTPUT ===\n"
            f"{json.dumps(ml_result, indent=2, default=str)}\n\n"
            "Interpret this quantitative analysis into structured investment findings.\n"
            "Source ID for all claims: 'src_price_history'.\n"
            "Call produce_analysis with your structured findings."
        )

        response = self._client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": QUANT_ANALYZE_SYSTEM},
                {"role": "user",   "content": user_message},
            ],
            tools=[PRODUCE_ANALYSIS_TOOL],
            tool_choice={"type": "function", "function": {"name": "produce_analysis"}},
            max_completion_tokens=4096,
        )

        message = response.choices[0].message
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "produce_analysis":
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as exc:
                        self.logger.error("Failed to parse produce_analysis JSON: %s", exc)
                        break

                    findings: list[AgentClaim] = [
                        AgentClaim(
                            text=str(f.get("text", "")),
                            source_ids=["src_price_history"],  # always our single source
                            confidence=float(max(0.0, min(1.0, f.get("confidence", 0.5)))),
                        )
                        for f in tool_input.get("findings", [])
                    ]

                    return AgentSubReport(
                        agent_name=self.agent_name,
                        findings=findings,
                        sources=[source],
                        confidence_score=float(
                            max(0.0, min(1.0, tool_input.get("confidence_score", 0.5)))
                        ),
                        conflicts=[],
                        raw_data_summary=str(
                            tool_input.get("raw_data_summary", "")
                        ),
                    )

        # ── Fallback: LLM didn't call the tool — build minimal report ──
        self.logger.error(
            "produce_analysis not returned for %s — using minimal fallback", ticker
        )
        pred    = ml_result.get("prediction", "UNKNOWN")
        proba   = ml_result.get("probabilities", {})
        acc     = ml_result.get("holdout_accuracy", 0.5)
        conf    = min(0.65, acc)

        return AgentSubReport(
            agent_name=self.agent_name,
            findings=[
                AgentClaim(
                    text=(
                        f"Random Forest model predicts {pred} price direction over the next "
                        f"20 trading days with {proba.get(pred, 0)*100:.1f}% probability "
                        f"(holdout accuracy: {acc*100:.1f}%). This is a technical momentum "
                        "signal only — not a fundamental valuation."
                    ),
                    source_ids=["src_price_history"],
                    confidence=conf,
                )
            ],
            sources=[source],
            confidence_score=conf,
            conflicts=[],
            raw_data_summary=(
                f"RandomForest trained on {ml_result.get('training_samples')} samples, "
                f"evaluated on {ml_result.get('holdout_samples')} holdout samples. "
                f"Holdout accuracy: {acc:.2%}."
            ),
        )

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _error_stub(self, reason: str) -> AgentSubReport:
        """Build a zero-confidence stub for cases where no data is available."""
        return AgentSubReport(
            agent_name=self.agent_name,
            findings=[],
            sources=[],
            confidence_score=0.0,
            conflicts=[],
            raw_data_summary=(
                f"Quantitative analysis unavailable: {reason} "
                "This agent requires a publicly traded stock with ≥120 trading days of history."
            ),
        )
