"""Company identity validation before running expensive LLM analysis."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from src.data.edgar import lookup_company_title

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNIVERSE_PATH = PROJECT_ROOT / "data" / "stock_universe.json"

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "llc", "plc", "sa", "ag", "nv", "group", "holdings",
}

COMMON_ALIASES: dict[str, list[str]] = {
    "GOOGL": ["Google"],
    "META": ["Facebook"],
    "DIS": ["Disney"],
    "XOM": ["Exxon", "Exxon Mobil"],
    "BRK-B": ["Berkshire"],
    "SQ": ["Square"],
    "KO": ["Coke"],
    "MCD": ["McDonalds", "McDonald's"],
    "PG": ["P&G"],
    "JNJ": ["J&J"],
    "T": ["AT and T", "AT&T"],
}


@dataclass(frozen=True)
class CompanyIdentity:
    """Verified company identity used by the due diligence pipeline."""

    company_name: str
    ticker: str | None
    confidence: float
    source: str


class CompanyValidationError(ValueError):
    """Raised when a requested company cannot be verified."""


def validate_company_identity(company_name: str, ticker: str | None = None) -> CompanyIdentity:
    """Verify that the requested company appears to be real before analysis.

    The validator intentionally favors precision over recall: it accepts companies
    from the curated stock universe, verifies public tickers via SEC EDGAR when
    needed, and rejects low-confidence names before the LLM can hallucinate a memo.

    Args:
        company_name: User-supplied company name.
        ticker: Optional stock ticker.

    Returns:
        A verified company identity, using a canonical company name when known.

    Raises:
        CompanyValidationError: If the name/ticker pair cannot be verified.
    """
    raw_name = (company_name or "").strip()
    raw_ticker = _normalize_ticker(ticker)

    if not raw_name and not raw_ticker:
        raise CompanyValidationError("Enter a company name or stock ticker to run analysis.")

    if raw_name and _looks_like_noise(raw_name):
        raise CompanyValidationError(
            f"Could not verify '{raw_name}' as a real company. "
            "Please enter an official company name and ticker."
        )

    universe = _load_universe()

    if raw_ticker:
        identity = _validate_by_ticker(raw_name, raw_ticker, universe)
        if identity:
            return identity
        raise CompanyValidationError(
            f"Could not verify ticker '{raw_ticker}'. "
            "Please check the ticker or use a company from the supported public universe."
        )

    identity = _validate_by_company_name(raw_name, universe)
    if identity:
        return identity

    raise CompanyValidationError(
        f"Could not verify '{raw_name}' as a real company. "
        "Please enter an official company name and ticker, e.g. 'Apple Inc' and 'AAPL'."
    )


def _validate_by_ticker(
    company_name: str,
    ticker: str,
    universe: list[dict],
) -> CompanyIdentity | None:
    """Validate an identity when the user supplied a ticker."""
    by_ticker = {str(s["ticker"]).upper(): s for s in universe}
    stock = by_ticker.get(ticker)

    if stock:
        official = str(stock["company"])
        aliases = COMMON_ALIASES.get(ticker, [])
        if _name_matches(company_name, ticker, official, aliases):
            return CompanyIdentity(official, ticker, confidence=0.95, source="stock_universe")
        raise CompanyValidationError(
            f"Ticker '{ticker}' belongs to {official}, but the company name "
            f"'{company_name}' does not appear to match."
        )

    try:
        sec_title = lookup_company_title(ticker)
    except Exception as exc:
        logger.warning("SEC ticker validation failed for %s: %s", ticker, exc)
        sec_title = None

    if sec_title and _name_matches(company_name, ticker, sec_title, COMMON_ALIASES.get(ticker, [])):
        return CompanyIdentity(sec_title, ticker, confidence=0.90, source="sec_edgar")

    return None


def _validate_by_company_name(company_name: str, universe: list[dict]) -> CompanyIdentity | None:
    """Validate an identity from the curated universe using fuzzy matching."""
    if not company_name:
        return None

    best_stock: dict | None = None
    best_score = 0.0
    for stock in universe:
        ticker = str(stock["ticker"]).upper()
        names = [str(stock["company"]), *COMMON_ALIASES.get(ticker, [])]
        score = max(_name_score(company_name, name) for name in names)
        if score > best_score:
            best_stock = stock
            best_score = score

    if best_stock and best_score >= 0.82:
        return CompanyIdentity(
            company_name=str(best_stock["company"]),
            ticker=str(best_stock["ticker"]).upper(),
            confidence=best_score,
            source="stock_universe",
        )
    return None


def _load_universe() -> list[dict]:
    """Load the local curated public-company universe."""
    try:
        with open(UNIVERSE_PATH) as f:
            return list(json.load(f).get("stocks", []))
    except FileNotFoundError:
        logger.warning("Stock universe not found at %s", UNIVERSE_PATH)
        return []


def _normalize_ticker(ticker: str | None) -> str | None:
    """Normalize ticker symbols while preserving hyphenated tickers."""
    if not ticker:
        return None
    cleaned = ticker.strip().upper().replace(".", "-")
    return cleaned or None


def _name_matches(company_name: str, ticker: str, official_name: str, aliases: list[str]) -> bool:
    """Return True when user input plausibly refers to the official company."""
    if not company_name:
        return True

    normalized_input = _normalize_name(company_name)
    if normalized_input == _normalize_name(ticker):
        return True

    return max(_name_score(company_name, name) for name in [official_name, *aliases]) >= 0.72


def _name_score(a: str, b: str) -> float:
    """Similarity score between two company names after normalization."""
    norm_a = _normalize_name(a)
    norm_b = _normalize_name(b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0
    if _safe_containment_match(norm_a, norm_b):
        return 0.90
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def _safe_containment_match(norm_a: str, norm_b: str) -> bool:
    """Allow short-name containment without accepting adversarial phrases."""
    if len(norm_a) < 4 or len(norm_b) < 4:
        return False
    if norm_a not in norm_b and norm_b not in norm_a:
        return False

    tokens_a = norm_a.split()
    tokens_b = norm_b.split()
    shorter, longer = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)

    # "Apple" should match "Apple Inc" after suffix removal, but
    # "Definitely Not Apple" should not match "Apple".
    return len(longer) - len(shorter) <= 1


def _normalize_name(value: str) -> str:
    """Normalize a company name for matching."""
    text = value.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    tokens = [token for token in re.split(r"[\s-]+", text) if token and token not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def _looks_like_noise(company_name: str) -> bool:
    """Detect obvious gibberish before doing any external lookup."""
    normalized = _normalize_name(company_name)
    letters = re.sub(r"[^a-z]", "", normalized)
    if len(letters) < 2:
        return True
    if len(letters) >= 8 and not re.search(r"[aeiouy]", letters):
        return True
    return False
