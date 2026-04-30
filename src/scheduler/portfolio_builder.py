"""Portfolio builder — converts weekly AI ratings into an actionable allocation.

Given an investment amount, this module:
    1. Loads the latest WeeklyReport (BUY / STRONG BUY rated stocks)
    2. Computes weights from: suggested_weight_pct × rating_boost × confidence
    3. Applies a max-position cap and normalises to 100 %
    4. Fetches live closing prices via yfinance
    5. Returns holdings (shares, $ amounts) + S&P 500 benchmark metrics
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

RATING_BOOST: dict[str, float] = {"STRONG BUY": 1.5, "BUY": 1.0}
BUY_RATINGS  = frozenset(RATING_BOOST)


# ─── Public entry point ───────────────────────────────────────────────────────

def build_portfolio(
    amount: float,
    max_positions: int = 10,
    max_position_pct: float = 30.0,
    strong_buy_only: bool = False,
) -> dict[str, Any]:
    """Build a portfolio allocation from the latest weekly report.

    Args:
        amount:           Total USD to invest.
        max_positions:    Maximum number of holdings (5–20).
        max_position_pct: Hard cap on any single position's weight (%).
        strong_buy_only:  If True, restrict to STRONG BUY rated stocks only.

    Returns:
        Dict with keys: holdings, sector_breakdown, sp500, summary stats.
        On error returns {'error': code, 'message': str}.
    """
    from src.scheduler.recommender import load_latest_report

    report = load_latest_report()
    if not report:
        return {
            "error": "no_report",
            "message": (
                "No weekly report found. "
                "Go to Weekly Rankings and generate one first."
            ),
        }

    eligible = {"STRONG BUY"} if strong_buy_only else BUY_RATINGS
    candidates = [r for r in report.ratings if r.rating in eligible]

    if not candidates:
        label = "STRONG BUY" if strong_buy_only else "BUY or STRONG BUY"
        return {
            "error": "no_candidates",
            "message": (
                f"No {label}-rated stocks in the latest report. "
                "Try regenerating the weekly report or broadening your filter."
            ),
        }

    # Trim to max_positions (ratings are already sorted best → worst)
    candidates = candidates[:max_positions]

    # ── Weight calculation ────────────────────────────────────────────────────
    raw = [
        r.suggested_weight_pct * RATING_BOOST.get(r.rating, 1.0) * r.confidence
        for r in candidates
    ]
    total_raw = sum(raw) or 1.0
    weights   = [w / total_raw * 100.0 for w in raw]
    weights   = _cap_and_normalise(weights, max_position_pct)

    # ── Live prices ───────────────────────────────────────────────────────────
    tickers = [r.ticker for r in candidates]
    prices  = _fetch_prices(tickers)

    # ── Build holdings ────────────────────────────────────────────────────────
    holdings: list[dict] = []
    total_invested = 0.0

    for i, rating in enumerate(candidates):
        price = prices.get(rating.ticker)
        if not price or price <= 0:
            logger.warning("Skipping %s — no price available", rating.ticker)
            continue

        dollar_amt = round(amount * weights[i] / 100.0, 2)
        shares     = round(dollar_amt / price, 4)
        total_invested += dollar_amt

        holdings.append({
            "ticker":        rating.ticker,
            "company":       rating.company_name,
            "sector":        rating.sector,
            "rating":        rating.rating,
            "rank":          rating.rank,
            "confidence":    round(rating.confidence, 4),
            "weight_pct":    round(weights[i], 2),
            "dollar_amount": dollar_amt,
            "shares":        shares,
            "current_price": round(price, 2),
            "bull_case":     rating.bull_case,
            "bear_case":     rating.bear_case,
            "rationale":     rating.rationale,
        })

    if not holdings:
        return {
            "error": "no_prices",
            "message": "Could not fetch current prices for any candidate. Try again shortly.",
        }

    # ── Sector breakdown ──────────────────────────────────────────────────────
    sector_breakdown: dict[str, float] = {}
    for h in holdings:
        sector_breakdown[h["sector"]] = round(
            sector_breakdown.get(h["sector"], 0.0) + h["weight_pct"], 2
        )
    # Sort by weight descending
    sector_breakdown = dict(
        sorted(sector_breakdown.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Portfolio-level stats ─────────────────────────────────────────────────
    avg_confidence = round(
        sum(h["confidence"] for h in holdings) / len(holdings), 4
    )
    strong_buy_count = sum(1 for h in holdings if h["rating"] == "STRONG BUY")

    # ── S&P 500 benchmark ─────────────────────────────────────────────────────
    sp500 = _fetch_sp500_metrics()

    return {
        "generated_at":     datetime.utcnow().isoformat(),
        "week_of":          report.week_of,
        "investment_amount": round(amount, 2),
        "total_invested":   round(total_invested, 2),
        "cash_remainder":   round(amount - total_invested, 2),
        "num_holdings":     len(holdings),
        "avg_confidence":   avg_confidence,
        "strong_buy_count": strong_buy_count,
        "holdings":         holdings,
        "sector_breakdown": sector_breakdown,
        "sp500":            sp500,
        "macro_commentary": getattr(report, "macro_commentary", ""),
        "universe_size":    report.universe_size,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _cap_and_normalise(weights: list[float], cap: float) -> list[float]:
    """Iteratively cap any weight exceeding `cap` %, redistributing excess."""
    w = weights[:]
    for _ in range(30):
        excess   = sum(max(0.0, x - cap) for x in w)
        if excess < 1e-8:
            break
        uncapped = [i for i, x in enumerate(w) if x < cap]
        if not uncapped:
            break
        per_item = excess / len(uncapped)
        for i in range(len(w)):
            if w[i] >= cap:
                w[i] = cap
            else:
                w[i] = min(cap, w[i] + per_item)
    # Final normalise in case of floating-point drift
    total = sum(w) or 1.0
    return [x / total * 100.0 for x in w]


def _fetch_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch latest closing prices for multiple tickers via yfinance."""
    try:
        import yfinance as yf

        data = yf.download(
            tickers,
            period="5d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data.empty:
            return {}

        close = data.get("Close", data)

        prices: dict[str, float] = {}
        if hasattr(close, "columns"):
            for ticker in tickers:
                if ticker in close.columns:
                    series = close[ticker].dropna()
                    if not series.empty:
                        prices[ticker] = float(series.iloc[-1])
        else:
            series = close.dropna()
            if not series.empty and tickers:
                prices[tickers[0]] = float(series.iloc[-1])

        return prices

    except Exception as exc:
        logger.error("Price fetch failed: %s", exc)
        return {}


def _fetch_sp500_metrics() -> dict[str, Any]:
    """Fetch S&P 500 1-month, 3-month, and YTD returns via yfinance."""
    try:
        import yfinance as yf

        hist = yf.Ticker("^GSPC").history(period="1y")
        if hist.empty:
            return {}

        close  = hist["Close"]
        latest = float(close.iloc[-1])

        def _ret(n: int) -> float | None:
            if len(close) < n:
                return None
            return round((latest / float(close.iloc[-n]) - 1) * 100, 2)

        # YTD: from first trading day of current calendar year
        this_year = close.index[-1].year
        ytd_series = close[close.index.year == this_year]
        ytd = (
            round((latest / float(ytd_series.iloc[0]) - 1) * 100, 2)
            if not ytd_series.empty
            else None
        )

        return {
            "name":         "S&P 500",
            "latest_close": round(latest, 2),
            "return_1w":    _ret(5),
            "return_1m":    _ret(21),
            "return_3m":    _ret(63),
            "return_ytd":   ytd,
        }

    except Exception as exc:
        logger.error("S&P 500 fetch failed: %s", exc)
        return {}
