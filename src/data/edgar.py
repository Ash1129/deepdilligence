"""SEC EDGAR API client for fetching company filings."""

import logging
from datetime import datetime
from typing import Any

import requests

from src.utils.cache import disk_cache
from src.utils.config import get_sec_edgar_user_agent

logger = logging.getLogger(__name__)

EDGAR_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_XBRL_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _get_headers() -> dict[str, str]:
    """Return headers required by SEC EDGAR (User-Agent with contact email)."""
    return {
        "User-Agent": get_sec_edgar_user_agent(),
        "Accept": "application/json",
    }


@disk_cache(subfolder="edgar")
def lookup_cik(ticker: str) -> str | None:
    """Look up the CIK number for a given stock ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL').

    Returns:
        Zero-padded 10-digit CIK string, or None if not found.
    """
    resp = requests.get(EDGAR_COMPANY_TICKERS_URL, headers=_get_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()

    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)

    logger.warning("Ticker %s not found in EDGAR company tickers", ticker)
    return None


@disk_cache(subfolder="edgar")
def get_company_facts(ticker: str) -> dict[str, Any] | None:
    """Fetch XBRL company facts (financial data) for a ticker.

    Returns structured financial data including revenue, assets, etc.
    """
    cik = lookup_cik(ticker)
    if not cik:
        return None

    url = EDGAR_XBRL_COMPANYFACTS_URL.format(cik=cik)
    resp = requests.get(url, headers=_get_headers(), timeout=15)
    resp.raise_for_status()
    return resp.json()


@disk_cache(subfolder="edgar")
def get_filings(ticker: str, filing_type: str = "10-K", count: int = 5) -> list[dict[str, Any]]:
    """Fetch recent filings metadata for a company.

    Args:
        ticker: Stock ticker symbol.
        filing_type: Filing type to filter (e.g., '10-K', '10-Q').
        count: Maximum number of filings to return.

    Returns:
        List of filing metadata dicts with keys: accessionNumber, filingDate,
        primaryDocument, form.
    """
    cik = lookup_cik(ticker)
    if not cik:
        return []

    url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=_get_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []

    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    results = []
    for i, form in enumerate(forms):
        if form == filing_type and len(results) < count:
            results.append({
                "accessionNumber": accession_numbers[i],
                "filingDate": filing_dates[i],
                "primaryDocument": primary_docs[i],
                "form": form,
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_numbers[i].replace('-', '')}/{primary_docs[i]}",
            })

    return results


@disk_cache(subfolder="edgar")
def get_revenue_data(ticker: str) -> list[dict[str, Any]]:
    """Extract revenue figures from XBRL company facts.

    Returns:
        List of dicts with keys: period, value, unit, filed_date.
    """
    facts = get_company_facts(ticker)
    if not facts:
        return []

    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    # Try common revenue field names
    revenue_fields = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]
    for field in revenue_fields:
        if field in us_gaap:
            units = us_gaap[field].get("units", {})
            usd_entries = units.get("USD", [])
            results = []
            for entry in usd_entries:
                if entry.get("form") in ("10-K", "10-Q"):
                    results.append({
                        "period": f"{entry.get('start', 'N/A')} to {entry.get('end', 'N/A')}",
                        "value": entry.get("val"),
                        "unit": "USD",
                        "filed_date": entry.get("filed", ""),
                        "form": entry.get("form", ""),
                    })
            if results:
                return results

    return []
