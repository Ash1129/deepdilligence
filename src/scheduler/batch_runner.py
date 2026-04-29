"""Batch runner — runs the DeepDiligence pipeline on every stock in the watchlist.

Caching means previously-analysed companies are returned instantly; only new
or stale entries make live API calls.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.models.schemas import InvestmentMemo
from src.orchestrator import Orchestrator
from src.utils.config import CACHE_DIR

logger = logging.getLogger(__name__)

WATCHLIST_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "watchlist.json"
MEMOS_DIR = CACHE_DIR / "memos"


def load_watchlist() -> list[dict]:
    """Load the watchlist from data/watchlist.json."""
    with open(WATCHLIST_PATH) as f:
        data = json.load(f)
    return data["stocks"]


def _memo_cache_path(ticker: str) -> Path:
    """Path for a cached memo JSON."""
    MEMOS_DIR.mkdir(parents=True, exist_ok=True)
    return MEMOS_DIR / f"{ticker.upper()}_memo.json"


def load_cached_memo(ticker: str) -> InvestmentMemo | None:
    """Load a previously saved memo from disk, or None if not found."""
    path = _memo_cache_path(ticker)
    if path.exists():
        try:
            with open(path) as f:
                return InvestmentMemo.model_validate(json.load(f))
        except Exception as e:
            logger.warning("Failed to load cached memo for %s: %s", ticker, e)
    return None


def save_memo(memo: InvestmentMemo, ticker: str) -> None:
    """Persist a memo to disk for reuse."""
    path = _memo_cache_path(ticker)
    with open(path, "w") as f:
        json.dump(memo.model_dump(mode="json"), f, indent=2)
    logger.info("Saved memo → %s", path.name)


def _run_one(stock: dict, force_refresh: bool) -> tuple[dict, InvestmentMemo | None]:
    """Run pipeline for a single stock. Returns (stock, memo)."""
    company = stock["company"]
    ticker  = stock["ticker"]

    if not force_refresh:
        cached = load_cached_memo(ticker)
        if cached:
            logger.info("Cache hit for %s — skipping pipeline run", ticker)
            return stock, cached

    logger.info("Running pipeline for %s (%s)…", company, ticker)
    try:
        memo = Orchestrator(company_name=company, ticker=ticker).run()
        save_memo(memo, ticker)
        return stock, memo
    except Exception as exc:
        logger.error("Pipeline failed for %s: %s", ticker, exc, exc_info=True)
        return stock, None


def run_batch(
    force_refresh: bool = False,
    max_workers: int = 3,
    on_complete: callable | None = None,
) -> dict[str, InvestmentMemo]:
    """Run (or load from cache) the pipeline for every stock in the watchlist.

    Args:
        force_refresh: If True, ignore cached memos and re-run all pipelines.
        max_workers: How many companies to analyse concurrently.
                     Keep low (2-3) to avoid OpenAI rate limits.
        on_complete: Optional callback(stock, memo) called after each company.

    Returns:
        Dict mapping ticker → InvestmentMemo for every successful run.
    """
    stocks = load_watchlist()
    results: dict[str, InvestmentMemo] = {}

    logger.info("Starting batch run for %d stocks (force_refresh=%s)", len(stocks), force_refresh)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="batch") as pool:
        futures = {pool.submit(_run_one, s, force_refresh): s for s in stocks}
        for future in as_completed(futures):
            stock, memo = future.result()
            ticker = stock["ticker"]
            if memo:
                results[ticker] = memo
                logger.info("✓ %s (%s) — %d findings", stock["company"], ticker, len(memo.sections))
            else:
                logger.warning("✗ %s (%s) — failed, skipping", stock["company"], ticker)
            if on_complete:
                on_complete(stock, memo)

    logger.info("Batch complete — %d / %d succeeded", len(results), len(stocks))
    return results
