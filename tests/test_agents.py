"""Tests for specialist agent schema handling and report construction."""

from __future__ import annotations

import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Set env vars before importing project modules
os.environ["OPENAI_API_KEY"] = "test-openai-key"
os.environ["NEWS_API_KEY"] = "test-news-api-key"
os.environ["SEC_EDGAR_USER_AGENT"] = "TestBot test@example.com"

from src.models.schemas import (
    AgentClaim,
    AgentSubReport,
    ConflictingClaim,
    Source,
    SourceType,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_source(id: str = "s1", url: str = "https://example.com") -> dict:
    return {
        "id": id,
        "url": url,
        "title": f"Source {id}",
        "snippet": "Some relevant excerpt from the source.",
        "source_type": "news_article",
    }


def _make_finding(text: str = "Revenue grew 20% YoY", source_ids: list[str] | None = None, confidence: float = 0.85) -> dict:
    return {
        "text": text,
        "source_ids": source_ids or ["s1"],
        "confidence": confidence,
    }


def _make_tool_output(
    agent_name: str = "financial_analyst",
    sources: list[dict] | None = None,
    findings: list[dict] | None = None,
    conflicts: list[dict] | None = None,
    confidence_score: float = 0.80,
    raw_data_summary: str = "Reviewed 10 EDGAR filings.",
) -> dict:
    return {
        "sources": sources or [_make_source()],
        "findings": findings or [_make_finding()],
        "conflicts": conflicts or [],
        "confidence_score": confidence_score,
        "raw_data_summary": raw_data_summary,
    }


# ─── ReactAgent._build_subreport tests ───────────────────────────────────────

class TestBuildSubreport:
    """Test ReactAgent._build_subreport() in isolation (no LLM calls)."""

    def _get_react_agent(self) -> "ReactAgent":
        from src.agents.react_agent import ReactAgent

        class ConcreteAgent(ReactAgent):
            @property
            def agent_name(self) -> str:
                return "test_agent"

            @property
            def description(self) -> str:
                return "Test agent"

            def gather_data(self) -> dict:
                return {}

            def analyze(self, raw_data: dict) -> AgentSubReport:
                return self._produce_structured_report("sys", raw_data)

        with patch("src.agents.react_agent.OpenAI"):
            return ConcreteAgent("TestCo", "TEST")

    def test_basic_subreport(self):
        agent = self._get_react_agent()
        tool_output = _make_tool_output()
        report = agent._build_subreport(tool_output)

        assert isinstance(report, AgentSubReport)
        assert report.agent_name == "test_agent"
        assert len(report.findings) == 1
        assert len(report.sources) == 1
        assert report.confidence_score == 0.80

    def test_source_id_validation(self):
        """Findings that reference non-existent source IDs should have those IDs dropped."""
        agent = self._get_react_agent()
        tool_output = _make_tool_output(
            sources=[_make_source(id="real_src")],
            findings=[_make_finding(source_ids=["real_src", "nonexistent_src"])],
        )
        report = agent._build_subreport(tool_output)
        assert report.findings[0].source_ids == ["real_src"]

    def test_confidence_clamping(self):
        """Confidence values outside [0,1] should be clamped."""
        agent = self._get_react_agent()
        tool_output = _make_tool_output(
            findings=[
                _make_finding(confidence=1.5),
                _make_finding(confidence=-0.2),
            ],
            confidence_score=2.0,
        )
        report = agent._build_subreport(tool_output)
        for finding in report.findings:
            assert 0.0 <= finding.confidence <= 1.0
        assert 0.0 <= report.confidence_score <= 1.0

    def test_invalid_source_type_defaults_to_other(self):
        """Invalid source_type strings should fall back to SourceType.OTHER."""
        agent = self._get_react_agent()
        tool_output = _make_tool_output(
            sources=[{
                "id": "s1",
                "url": "https://example.com",
                "title": "A source",
                "snippet": "...",
                "source_type": "totally_invalid_type",
            }]
        )
        report = agent._build_subreport(tool_output)
        assert report.sources[0].source_type == SourceType.OTHER

    def test_conflict_parsing(self):
        """ConflictingClaims should be parsed from tool output conflicts."""
        agent = self._get_react_agent()
        tool_output = _make_tool_output(
            conflicts=[{
                "claim_a_text": "Revenue is growing",
                "claim_b_text": "Revenue is declining",
                "description": "Contradictory signals in Q4",
            }]
        )
        report = agent._build_subreport(tool_output)
        assert len(report.conflicts) == 1
        assert "growing" in report.conflicts[0].claim_a.text
        assert "declining" in report.conflicts[0].claim_b.text

    def test_empty_tool_output(self):
        """Empty tool output should produce a valid zero-findings report."""
        agent = self._get_react_agent()
        report = agent._build_subreport({})
        assert isinstance(report, AgentSubReport)
        assert report.findings == []
        assert report.sources == []
        assert report.confidence_score == 0.5  # default

    def test_snippet_truncation(self):
        """Source snippets longer than 500 chars should be truncated."""
        agent = self._get_react_agent()
        long_snippet = "x" * 1000
        tool_output = _make_tool_output(
            sources=[{
                "id": "s1",
                "url": "https://example.com",
                "title": "Long source",
                "snippet": long_snippet,
                "source_type": "news_article",
            }]
        )
        report = agent._build_subreport(tool_output)
        assert len(report.sources[0].snippet) <= 500


# ─── Agent name and description properties ───────────────────────────────────

class TestAgentProperties:
    """Verify each specialist agent has the correct agent_name and description."""

    def _make_agent(self, cls_path: str, *args, **kwargs):
        module_path, class_name = cls_path.rsplit(".", 1)
        with patch("src.agents.react_agent.OpenAI"):
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls(*args, **kwargs)

    def test_financial_agent_name(self):
        agent = self._make_agent("src.agents.financial.FinancialAgent", "Apple Inc", "AAPL")
        assert agent.agent_name == "financial_analyst"

    def test_team_agent_name(self):
        agent = self._make_agent("src.agents.team.TeamAgent", "Apple Inc", "AAPL")
        assert agent.agent_name == "team_culture"

    def test_market_agent_name(self):
        agent = self._make_agent("src.agents.market.MarketAgent", "Apple Inc", "AAPL")
        assert agent.agent_name == "market_competitive"

    def test_risk_agent_name(self):
        agent = self._make_agent("src.agents.risk.RiskAgent", "Apple Inc", "AAPL")
        assert agent.agent_name == "risk_sentiment"

    def test_all_agents_have_description(self):
        from src.agents.financial import FinancialAgent
        from src.agents.market import MarketAgent
        from src.agents.risk import RiskAgent
        from src.agents.team import TeamAgent

        with patch("src.agents.react_agent.OpenAI"):
            agents = [
                FinancialAgent("Co", "CO"),
                TeamAgent("Co", "CO"),
                MarketAgent("Co", "CO"),
                RiskAgent("Co", "CO"),
            ]
        for agent in agents:
            assert isinstance(agent.description, str)
            assert len(agent.description) > 10


# ─── Cache key uniqueness ─────────────────────────────────────────────────────

class TestCacheKey:
    def test_different_companies_different_keys(self):
        from src.agents.financial import FinancialAgent
        with patch("src.agents.react_agent.OpenAI"):
            a1 = FinancialAgent("Apple Inc", "AAPL")
            a2 = FinancialAgent("Microsoft", "MSFT")
        assert a1._cache_key() != a2._cache_key()

    def test_same_company_same_key(self):
        from src.agents.financial import FinancialAgent
        with patch("src.agents.react_agent.OpenAI"):
            a1 = FinancialAgent("Apple Inc", "AAPL")
            a2 = FinancialAgent("Apple Inc", "AAPL")
        assert a1._cache_key() == a2._cache_key()

    def test_different_agents_different_keys(self):
        from src.agents.financial import FinancialAgent
        from src.agents.risk import RiskAgent
        with patch("src.agents.react_agent.OpenAI"):
            fa = FinancialAgent("Apple Inc", "AAPL")
            ra = RiskAgent("Apple Inc", "AAPL")
        # Same company, different agent types → different cache keys
        assert fa._cache_key() != ra._cache_key()


# ─── Orchestrator unit tests ──────────────────────────────────────────────────

class TestOrchestrator:
    """Test Orchestrator logic without running real agents."""

    def _make_stub_report(self, agent_name: str, n_findings: int = 3) -> AgentSubReport:
        return AgentSubReport(
            agent_name=agent_name,
            findings=[
                AgentClaim(text=f"Finding {i}", source_ids=[], confidence=0.8)
                for i in range(n_findings)
            ],
            sources=[],
            confidence_score=0.75,
            conflicts=[],
            raw_data_summary="Stub report.",
        )

    def test_error_subreport(self):
        from src.orchestrator import Orchestrator
        report = Orchestrator._error_subreport("test_agent", ValueError("boom"))
        assert report.agent_name == "test_agent"
        assert report.confidence_score == 0.0
        assert report.findings == []
        assert "ValueError" in report.raw_data_summary

    def test_orchestrator_instantiation(self):
        """Orchestrator should instantiate without hitting real APIs."""
        with patch("src.agents.synthesis.OpenAI"):
            from src.orchestrator import Orchestrator
            orch = Orchestrator("TestCo", ticker="TC")
            assert orch.company_name == "TestCo"
            assert orch.ticker == "TC"
