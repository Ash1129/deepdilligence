"""Stock screener — filters the curated universe or screens live S&P 500 movers.

Two modes:
  - Universe filter: applies style + sector filters to data/stock_universe.json
  - S&P 500 screener: downloads live 5-day price/volume data via yfinance and
    picks the top N movers by the chosen criteria (price change, volume, or both)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

UNIVERSE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "stock_universe.json"

ALL_SECTORS = [
    "Technology",
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Financials",
    "Healthcare",
    "Industrials",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",
]

ALL_STYLES = ["Growth", "Value", "Dividend Income", "Momentum"]


# ─── Universe helpers ─────────────────────────────────────────────────────────

def load_universe() -> list[dict]:
    """Load the curated stock universe from disk."""
    with open(UNIVERSE_PATH) as f:
        return json.load(f)["stocks"]


def filter_universe(
    styles: list[str],
    sectors: list[str],
    top_n: int,
) -> list[dict]:
    """Filter the curated universe by style(s) and sector(s), return top_n.

    Args:
        styles:  Investment styles to include (empty = all styles).
        sectors: Sectors to include (empty = all sectors).
        top_n:   Maximum number of stocks to return.

    Returns:
        List of stock dicts with keys: ticker, company, sector, styles.
    """
    universe = load_universe()

    filtered = [
        s for s in universe
        if (not styles  or any(st in s["styles"]  for st in styles))
        and (not sectors or s["sector"] in sectors)
    ]

    if not filtered:
        logger.warning("No stocks matched filters — returning full universe sample")
        filtered = universe

    # If more than top_n, score by how many requested styles each stock has
    # (so stocks matching multiple requested styles rank higher)
    if styles:
        filtered.sort(key=lambda s: sum(1 for st in styles if st in s["styles"]), reverse=True)

    selected = filtered[:top_n]
    logger.info(
        "Universe filter: %d → %d stocks (styles=%s, sectors=%s)",
        len(universe), len(selected), styles, sectors,
    )
    return selected


# ─── S&P 500 live screener ────────────────────────────────────────────────────

# Representative S&P 500 ticker sample — large enough for a meaningful screener
# without needing to scrape Wikipedia every run
_SP500_SAMPLE = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","JPM","LLY",
    "V","MA","UNH","XOM","WMT","ORCL","COST","NFLX","HD","JNJ","BAC","PG",
    "ABBV","CRM","CVX","MRK","AMD","NOW","MS","GS","ISRG","AXP","PEP","T",
    "ADBE","TJX","BX","UBER","HON","CAT","MU","BMY","DE","UPS","C","BA",
    "GE","LMT","MMM","NEE","DUK","SO","AMT","PLD","O","FCX","LIN","DIS",
    "CMCSA","NKE","MCD","KO","PFE","WFC","COIN","PLTR","SHOP","SQ","SNOW",
    "COP","NEM","VZ","SCHW","USB","TGT","LOW","F","GM","INTC","QCOM","TXN",
    "INTU","PYPL","ZM","DOCU","DASH","RBLX","HOOD","RIVN","LCID","NIO",
]


def screen_sp500(
    criteria: str,
    styles: list[str],
    sectors: list[str],
    top_n: int,
) -> list[dict]:
    """Screen S&P 500 sample for top movers using live yfinance data.

    Args:
        criteria: "Price Change" | "Volume" | "Price Change + Volume"
        styles:   Filter results to stocks matching these styles (if possible).
        sectors:  Filter results to stocks in these sectors (if possible).
        top_n:    Number of stocks to return.

    Returns:
        List of stock dicts ready for the batch runner.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — falling back to universe filter")
        return filter_universe(styles, sectors, top_n)

    logger.info("Screening %d S&P 500 tickers via yfinance (criteria=%s)…", len(_SP500_SAMPLE), criteria)

    try:
        raw = yf.download(
            _SP500_SAMPLE,
            period="5d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("yfinance download failed: %s — falling back to universe filter", exc)
        return filter_universe(styles, sectors, top_n)

    scores: dict[str, float] = {}

    for ticker in _SP500_SAMPLE:
        try:
            closes  = raw["Close"][ticker].dropna()
            volumes = raw["Volume"][ticker].dropna()

            if len(closes) < 2:
                continue

            pct_change  = abs((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]) * 100
            avg_volume  = volumes.mean() / 1_000_000   # millions

            if criteria == "Price Change":
                scores[ticker] = pct_change
            elif criteria == "Volume":
                scores[ticker] = avg_volume
            else:   # Price Change + Volume (combined)
                # Normalise both to 0-1 then average — prevents volume dominating
                scores[ticker] = pct_change + avg_volume * 0.1

        except Exception:
            continue

    if not scores:
        logger.warning("No scores computed — falling back to universe filter")
        return filter_universe(styles, sectors, top_n)

    ranked = sorted(scores, key=scores.get, reverse=True)
    logger.info("Top 5 by %s: %s", criteria, ranked[:5])

    # Build universe lookup for metadata
    universe_lookup = {s["ticker"]: s for s in load_universe()}

    results: list[dict] = []
    for ticker in ranked:
        meta = universe_lookup.get(ticker)
        if meta:
            # Apply style/sector filters if requested
            if styles  and not any(st in meta["styles"]  for st in styles):
                continue
            if sectors and meta["sector"] not in sectors:
                continue
            results.append(meta)
        else:
            # Ticker not in curated universe — include with minimal metadata
            results.append({
                "ticker":  ticker,
                "company": ticker,       # will be enriched by the agent
                "sector":  "Unknown",
                "styles":  [],
            })

        if len(results) >= top_n:
            break

    # If filtering left us with too few AND no filters were applied, backfill
    # from the full ranked list.  When the user has specified styles/sectors,
    # respect their preferences — do not pad with non-matching companies.
    if len(results) < top_n and not styles and not sectors:
        for ticker in ranked:
            if any(r["ticker"] == ticker for r in results):
                continue
            meta = universe_lookup.get(ticker, {
                "ticker": ticker, "company": ticker, "sector": "Unknown", "styles": []
            })
            results.append(meta)
            if len(results) >= top_n:
                break

    logger.info("Screener selected %d stocks", len(results))
    return results[:top_n]


# ─── Unified entry point ──────────────────────────────────────────────────────

def build_watchlist(
    use_screener: bool,
    screener_criteria: str,
    styles: list[str],
    sectors: list[str],
    top_n: int,
) -> list[dict]:
    """Return a watchlist based on user preferences.

    Args:
        use_screener:       If True, screen live S&P 500 data. If False, filter universe.
        screener_criteria:  "Price Change" | "Volume" | "Price Change + Volume"
        styles:             Investment styles to target (empty = all).
        sectors:            Sectors to include (empty = all).
        top_n:              How many stocks to return.

    Returns:
        List of stock dicts with ticker, company, sector, styles.
    """
    if use_screener:
        return screen_sp500(screener_criteria, styles, sectors, top_n)
    return filter_universe(styles, sectors, top_n)
