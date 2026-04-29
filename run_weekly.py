"""DeepDiligence — Weekly autonomous analysis runner.

Run manually:
    python3.11 run_weekly.py

Schedule via cron (every Monday 7am):
    0 7 * * 1 cd /path/to/deepdiligence && python3.11 run_weekly.py >> logs/weekly.log 2>&1

Args:
    --refresh   Force re-run all companies even if cached (default: use cache)
    --dry-run   Load existing memos and regenerate recommendations only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("weekly")

from src.scheduler.batch_runner import load_watchlist, run_batch, load_cached_memo
from src.scheduler.recommender import RecommendationEngine, save_report, load_latest_report
from src.models.schemas import InvestmentMemo


RATING_EMOJI = {
    "STRONG BUY":  "🟢🟢",
    "BUY":         "🟢",
    "HOLD":        "🟡",
    "SELL":        "🔴",
    "STRONG SELL": "🔴🔴",
}


def print_report(report) -> None:
    """Print a formatted weekly report to stdout."""
    print()
    print("=" * 70)
    print(f"  📈 DeepDiligence Weekly Report — {report.week_of}")
    print(f"  Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Universe:  {report.universe_size} companies analysed")
    print("=" * 70)

    print(f"\n📊 MACRO VIEW\n{report.macro_commentary}")

    print("\n🏭 SECTOR VIEWS")
    for sector, view in report.sector_views.items():
        arrow = "↑" if "Bullish" in view else ("↓" if "Bearish" in view else "→")
        print(f"  {arrow}  {sector}: {view}")

    print("\n📋 FULL RANKINGS")
    print(f"  {'#':<4} {'Ticker':<8} {'Rating':<14} {'Wt%':<6} {'Conf':<6} Company")
    print(f"  {'-'*4} {'-'*8} {'-'*14} {'-'*6} {'-'*6} {'-'*20}")
    for r in report.ratings:
        emoji = RATING_EMOJI.get(r.rating, "  ")
        print(
            f"  {r.rank:<4} {r.ticker:<8} {emoji} {r.rating:<12} "
            f"{r.suggested_weight_pct:>4.1f}%  {r.confidence:.0%}   {r.company_name}"
        )

    print(f"\n✅ TOP PICKS:  {', '.join(report.top_picks)}")
    if report.avoid:
        print(f"❌ AVOID:      {', '.join(report.avoid)}")

    print("\n📝 RATIONALES")
    for r in report.ratings:
        emoji = RATING_EMOJI.get(r.rating, "")
        print(f"\n  {emoji} {r.company_name} ({r.ticker}) — {r.rating}")
        print(f"  Bull: {r.bull_case}")
        print(f"  Bear: {r.bear_case}")
        print(f"  Why:  {r.rationale}")

    print()
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepDiligence weekly runner")
    parser.add_argument("--refresh",  action="store_true", help="Force re-run all companies")
    parser.add_argument("--dry-run",  action="store_true", help="Use existing memos only, skip pipeline")
    args = parser.parse_args()

    watchlist = load_watchlist()
    logger.info("Watchlist: %d companies", len(watchlist))

    # ── Step 1: collect memos ────────────────────────────────────────────────
    if args.dry_run:
        logger.info("Dry-run mode — loading cached memos only")
        memos: dict[str, InvestmentMemo] = {}
        for stock in watchlist:
            memo = load_cached_memo(stock["ticker"])
            if memo:
                memos[stock["ticker"]] = memo
                logger.info("  Loaded %s", stock["ticker"])
            else:
                logger.warning("  No cache for %s — skipping", stock["ticker"])
    else:
        def on_complete(stock, memo):
            status = "✓" if memo else "✗"
            logger.info("%s %s (%s)", status, stock["company"], stock["ticker"])

        memos = run_batch(
            force_refresh=args.refresh,
            max_workers=2,          # conservative to avoid rate limits
            on_complete=on_complete,
        )

    if not memos:
        logger.error("No memos available — aborting")
        sys.exit(1)

    logger.info("Memos ready: %d companies", len(memos))

    # ── Step 2: generate recommendations ────────────────────────────────────
    logger.info("Generating recommendations…")
    engine = RecommendationEngine()
    report = engine.generate(memos)

    # ── Step 3: save + print ─────────────────────────────────────────────────
    path = save_report(report)
    logger.info("Report saved → %s", path.name)

    print_report(report)


if __name__ == "__main__":
    main()
