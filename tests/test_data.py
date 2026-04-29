"""Tests for data pipeline clients."""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import responses

from src.models.schemas import (
    AgentClaim,
    AgentSubReport,
    ConflictingClaim,
    InvestmentMemo,
    Source,
    SourceType,
    SynthesizedSection,
)

# Set test env vars before importing modules that read config
os.environ["SEC_EDGAR_USER_AGENT"] = "TestBot test@example.com"
os.environ["NEWS_API_KEY"] = "test-news-api-key"
os.environ["OPENAI_API_KEY"] = "test-openai-key"

from src.data.edgar import (
    EDGAR_COMPANY_TICKERS_URL,
    EDGAR_SUBMISSIONS_URL,
    EDGAR_XBRL_COMPANYFACTS_URL,
    get_company_facts,
    get_filings,
    get_revenue_data,
    lookup_cik,
)
from src.data.jobs import _looks_like_job_title, scrape_careers_page
from src.data.news import NEWSAPI_EVERYTHING_URL, fetch_company_news
from src.data.web import scrape_url
from src.utils.cache import CACHE_DIR, clear_cache, disk_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def use_temp_cache(tmp_path, monkeypatch):
    """Redirect all caching to a temp directory for test isolation."""
    monkeypatch.setattr("src.utils.cache.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("src.utils.config.CACHE_DIR", tmp_path / "cache")


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_source_creation(self):
        source = Source(
            id="src-1",
            url="https://example.com",
            title="Test Source",
            snippet="Some excerpt",
            source_type=SourceType.NEWS_ARTICLE,
        )
        assert source.id == "src-1"
        assert source.source_type == SourceType.NEWS_ARTICLE
        assert isinstance(source.retrieved_at, datetime)

    def test_agent_claim_confidence_bounds(self):
        claim = AgentClaim(text="Revenue grew 20%", source_ids=["src-1"], confidence=0.85)
        assert claim.confidence == 0.85

        with pytest.raises(Exception):
            AgentClaim(text="Bad", source_ids=[], confidence=1.5)

        with pytest.raises(Exception):
            AgentClaim(text="Bad", source_ids=[], confidence=-0.1)

    def test_agent_sub_report(self):
        report = AgentSubReport(
            agent_name="financial_analyst",
            findings=[AgentClaim(text="Test claim", source_ids=[], confidence=0.9)],
            sources=[],
            confidence_score=0.8,
        )
        assert report.agent_name == "financial_analyst"
        assert len(report.findings) == 1

    def test_investment_memo(self):
        memo = InvestmentMemo(
            company_name="Test Corp",
            executive_summary="Looks good.",
            sections=[
                SynthesizedSection(
                    title="Financials",
                    content="Strong revenue growth.",
                    confidence_score=0.9,
                )
            ],
            overall_confidence=0.85,
        )
        assert memo.company_name == "Test Corp"
        assert len(memo.sections) == 1
        assert isinstance(memo.generated_at, datetime)

    def test_conflicting_claim(self):
        cc = ConflictingClaim(
            claim_a=AgentClaim(text="Revenue up", source_ids=["s1"], confidence=0.9),
            claim_b=AgentClaim(text="Revenue down", source_ids=["s2"], confidence=0.7),
            description="Contradictory revenue signals",
        )
        assert "up" in cc.claim_a.text
        assert "down" in cc.claim_b.text


# ---------------------------------------------------------------------------
# Cache Tests
# ---------------------------------------------------------------------------

class TestCache:
    def test_disk_cache_hit_and_miss(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.utils.cache.CACHE_DIR", tmp_path)
        call_count = 0

        @disk_cache(subfolder="test")
        def expensive_fn(x: int) -> dict:
            nonlocal call_count
            call_count += 1
            return {"result": x * 2}

        result1 = expensive_fn(5)
        assert result1 == {"result": 10}
        assert call_count == 1

        result2 = expensive_fn(5)
        assert result2 == {"result": 10}
        assert call_count == 1  # Cache hit — not called again

        result3 = expensive_fn(6)
        assert result3 == {"result": 12}
        assert call_count == 2  # Different args — cache miss

    def test_clear_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.utils.cache.CACHE_DIR", tmp_path)

        @disk_cache(subfolder="cleartest")
        def fn(x: int) -> dict:
            return {"v": x}

        fn(1)
        fn(2)
        assert len(list((tmp_path / "cleartest").glob("*.json"))) == 2

        deleted = clear_cache("cleartest")
        assert deleted == 2
        assert len(list((tmp_path / "cleartest").glob("*.json"))) == 0


# ---------------------------------------------------------------------------
# EDGAR Tests
# ---------------------------------------------------------------------------

MOCK_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
}


class TestEdgar:
    @responses.activate
    def test_lookup_cik(self):
        responses.add(responses.GET, EDGAR_COMPANY_TICKERS_URL, json=MOCK_TICKERS, status=200)
        cik = lookup_cik("AAPL")
        assert cik == "0000320193"

    @responses.activate
    def test_lookup_cik_not_found(self):
        responses.add(responses.GET, EDGAR_COMPANY_TICKERS_URL, json=MOCK_TICKERS, status=200)
        cik = lookup_cik("ZZZZ")
        assert cik is None

    @responses.activate
    def test_get_filings(self):
        responses.add(responses.GET, EDGAR_COMPANY_TICKERS_URL, json=MOCK_TICKERS, status=200)

        mock_submissions = {
            "filings": {
                "recent": {
                    "form": ["10-K", "10-Q", "10-K", "8-K"],
                    "accessionNumber": ["0001-1", "0001-2", "0001-3", "0001-4"],
                    "filingDate": ["2024-01-01", "2024-04-01", "2023-01-01", "2024-06-01"],
                    "primaryDocument": ["doc1.htm", "doc2.htm", "doc3.htm", "doc4.htm"],
                }
            }
        }
        responses.add(
            responses.GET,
            EDGAR_SUBMISSIONS_URL.format(cik="0000320193"),
            json=mock_submissions,
            status=200,
        )

        filings = get_filings("AAPL", filing_type="10-K", count=2)
        assert len(filings) == 2
        assert filings[0]["form"] == "10-K"
        assert filings[0]["filingDate"] == "2024-01-01"

    @responses.activate
    def test_get_revenue_data(self):
        responses.add(responses.GET, EDGAR_COMPANY_TICKERS_URL, json=MOCK_TICKERS, status=200)

        mock_facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"start": "2023-01-01", "end": "2023-12-31", "val": 100000000, "form": "10-K", "filed": "2024-02-01"},
                                {"start": "2024-01-01", "end": "2024-03-31", "val": 30000000, "form": "10-Q", "filed": "2024-05-01"},
                            ]
                        }
                    }
                }
            }
        }
        responses.add(
            responses.GET,
            EDGAR_XBRL_COMPANYFACTS_URL.format(cik="0000320193"),
            json=mock_facts,
            status=200,
        )

        revenue = get_revenue_data("AAPL")
        assert len(revenue) == 2
        assert revenue[0]["value"] == 100000000
        assert revenue[0]["unit"] == "USD"


# ---------------------------------------------------------------------------
# News Tests
# ---------------------------------------------------------------------------

class TestNews:
    @responses.activate
    def test_fetch_company_news(self):
        mock_response = {
            "status": "ok",
            "totalResults": 2,
            "articles": [
                {
                    "title": "Apple Reports Record Q4",
                    "description": "Strong earnings beat expectations.",
                    "url": "https://news.example.com/apple-q4",
                    "source": {"name": "TechNews"},
                    "publishedAt": "2024-10-30T12:00:00Z",
                    "content": "Apple reported record revenue...",
                },
                {
                    "title": "Apple Launches New Product",
                    "description": "New product line announced.",
                    "url": "https://news.example.com/apple-product",
                    "source": {"name": "Gadgets Daily"},
                    "publishedAt": "2024-10-28T08:00:00Z",
                    "content": "Apple unveiled...",
                },
            ],
        }
        responses.add(responses.GET, NEWSAPI_EVERYTHING_URL, json=mock_response, status=200)

        articles = fetch_company_news("Apple", days_back=30, page_size=10)
        assert len(articles) == 2
        assert articles[0]["title"] == "Apple Reports Record Q4"
        assert articles[0]["source"] == "TechNews"

    @responses.activate
    def test_fetch_company_news_error(self):
        mock_response = {"status": "error", "message": "rateLimited"}
        responses.add(responses.GET, NEWSAPI_EVERYTHING_URL, json=mock_response, status=200)

        articles = fetch_company_news("Apple")
        assert articles == []


# ---------------------------------------------------------------------------
# Web Scraper Tests
# ---------------------------------------------------------------------------

class TestWebScraper:
    @responses.activate
    def test_scrape_url_success(self):
        html = """
        <html>
        <head><title>About Us</title></head>
        <body>
            <nav>Navigation links</nav>
            <main>
                <h1>About Our Company</h1>
                <p>We build great products.</p>
                <p>Founded in 2020.</p>
            </main>
            <footer>Footer stuff</footer>
        </body>
        </html>
        """
        responses.add(responses.GET, "https://example.com/about", body=html, status=200)

        result = scrape_url("https://example.com/about")
        assert result["error"] is None
        assert result["title"] == "About Us"
        assert "great products" in result["text"]
        # Nav and footer should be stripped
        assert "Navigation links" not in result["text"]
        assert "Footer stuff" not in result["text"]

    @responses.activate
    def test_scrape_url_failure(self):
        responses.add(responses.GET, "https://example.com/404", status=404)
        result = scrape_url("https://example.com/404")
        assert result["text"] is None
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# Jobs Scraper Tests
# ---------------------------------------------------------------------------

class TestJobsScraper:
    @responses.activate
    def test_scrape_careers_page(self):
        html = """
        <html>
        <body>
            <h1>Careers</h1>
            <div class="jobs">
                <a href="/jobs/1">Senior Software Engineer</a>
                <a href="/jobs/2">Product Manager - Growth</a>
                <a href="/jobs/3">Data Scientist</a>
                <a href="/about">About Us</a>
            </div>
            <div class="departments">
                <span>Engineering</span>
                <span>Product</span>
                <span>Data</span>
            </div>
        </body>
        </html>
        """
        responses.add(responses.GET, "https://example.com/careers", body=html, status=200)

        result = scrape_careers_page("https://example.com/careers")
        assert result["error"] is None
        assert result["job_count"] == 3
        assert "Senior Software Engineer" in result["job_titles"]
        assert "Product Manager - Growth" in result["job_titles"]
        assert "Data Scientist" in result["job_titles"]

    def test_looks_like_job_title(self):
        assert _looks_like_job_title("Senior Software Engineer") is True
        assert _looks_like_job_title("Product Manager") is True
        assert _looks_like_job_title("About Us") is False
        assert _looks_like_job_title("") is False
        assert _looks_like_job_title("OK") is False

    @responses.activate
    def test_scrape_careers_page_failure(self):
        responses.add(responses.GET, "https://example.com/careers", status=500)
        result = scrape_careers_page("https://example.com/careers")
        assert result["error"] is not None
        assert result["job_count"] == 0
