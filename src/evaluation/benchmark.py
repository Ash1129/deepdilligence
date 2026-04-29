"""Benchmark dataset loader and runner for evaluating DeepDiligence memos."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.utils.config import BENCHMARKS_DIR

logger = logging.getLogger(__name__)


# ─── Benchmark data model ─────────────────────────────────────────────────────

class BenchmarkCompany(BaseModel):
    """Ground-truth profile for a benchmark company used in evaluation."""

    company_name: str = Field(..., description="Display name of the company")
    ticker: str | None = Field(None, description="Stock ticker (None for private companies)")
    sector: str = Field(..., description="Industry sector")
    description: str = Field(..., description="Brief company description (1-2 sentences)")

    # Ground truth facts the memo SHOULD surface
    known_facts: list[str] = Field(
        default_factory=list,
        description="Verifiable facts that a complete memo should mention",
    )
    known_risks: list[str] = Field(
        default_factory=list,
        description="Known risk signals the memo should flag",
    )
    known_strengths: list[str] = Field(
        default_factory=list,
        description="Known competitive strengths the memo should recognize",
    )

    # Expected high-level assessment (for calibration only — not a buy/sell recommendation)
    expected_sentiment: str = Field(
        ...,
        description="Expected overall tone: 'very_positive' | 'positive' | 'neutral' | 'negative' | 'very_negative'",
    )
    min_expected_confidence: float = Field(
        0.5,
        description="Floor for overall_confidence — large public companies should score high",
    )

    # Optional extra labels for domain-specific evaluation
    tags: list[str] = Field(default_factory=list, description="e.g. ['big_tech', 'high_growth', 'regulated']")


# ─── Loader ──────────────────────────────────────────────────────────────────

def load_benchmark(name: str) -> BenchmarkCompany:
    """Load a benchmark profile by company name or ticker.

    Searches `data/benchmarks/` for a JSON file named `{name_lower}.json`
    or `{ticker_lower}.json`.

    Args:
        name: Company name (e.g. "Apple") or ticker (e.g. "AAPL").

    Returns:
        Parsed BenchmarkCompany.

    Raises:
        FileNotFoundError: If no matching benchmark file exists.
    """
    candidates = [
        BENCHMARKS_DIR / f"{name.lower().replace(' ', '_')}.json",
        BENCHMARKS_DIR / f"{name.lower()}.json",
        BENCHMARKS_DIR / f"{name.upper()}.json",
    ]
    for path in candidates:
        if path.exists():
            logger.info("Loading benchmark from %s", path.name)
            with open(path) as f:
                return BenchmarkCompany.model_validate(json.load(f))

    raise FileNotFoundError(
        f"No benchmark file found for '{name}'. "
        f"Looked in: {[str(p) for p in candidates]}"
    )


def list_benchmarks() -> list[BenchmarkCompany]:
    """Load and return all available benchmark profiles."""
    profiles: list[BenchmarkCompany] = []
    for path in sorted(BENCHMARKS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                profiles.append(BenchmarkCompany.model_validate(json.load(f)))
        except Exception as exc:
            logger.warning("Failed to load benchmark %s: %s", path.name, exc)
    return profiles


def get_benchmark_names() -> list[str]:
    """Return a list of available benchmark company names."""
    return [b.company_name for b in list_benchmarks()]
