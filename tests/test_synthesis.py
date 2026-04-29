"""Tests for the SynthesisAgent and evaluation framework."""

from __future__ import annotations

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

os.environ["OPENAI_API_KEY"] = "test-openai-key"
os.environ["NEWS_API_KEY"] = "test-news-api-key"
os.environ["SEC_EDGAR_USER_AGENT"] = "TestBot test@example.com"

from src.evaluation.faithfulness import (
    FaithfulnessResult,
    SectionFaithfulness,
    faithfulness_report_text,
    score_faithfulness,
)
from src.evaluation.metrics import EvalResult, _check_coverage, _memo_text, compute_metrics
from src.models.schemas import (
    AgentClaim,
    AgentSubReport,
    ConflictingClaim,
    InvestmentMemo,
    Source,
    SourceType,
    SynthesizedSection,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_source(id: str = "s1") -> Source:
    return Source(
        id=id,
        url=f"https://example.com/{id}",
        title=f"Source {id}",
        snippet="Relevant excerpt.",
        source_type=SourceType.NEWS_ARTICLE,
    )


def _make_claim(text: str, source_ids: list[str], confidence: float = 0.8) -> AgentClaim:
    return AgentClaim(text=text, source_ids=source_ids, confidence=confidence)


def _make_section(
    title: str = "Test Section",
    claims: list[AgentClaim] | None = None,
    conflicts: list[ConflictingClaim] | None = None,
    confidence: float = 0.75,
) -> SynthesizedSection:
    return SynthesizedSection(
        title=title,
        content=f"Detailed analysis of {title}.",
        claims=claims or [],
        confidence_score=confidence,
        conflicting_claims=conflicts or [],
    )


def _make_memo(
    company: str = "TestCorp",
    sections: list[SynthesizedSection] | None = None,
    overall_confidence: float = 0.72,
    metadata: dict | None = None,
) -> InvestmentMemo:
    return InvestmentMemo(
        company_name=company,
        executive_summary="TestCorp shows strong fundamentals with some regulatory risk.",
        sections=sections or [],
        overall_confidence=overall_confidence,
        metadata=metadata or {"total_findings": 10, "total_sources": 8},
    )


# ─── Faithfulness tests ───────────────────────────────────────────────────────

class TestFaithfulness:
    def test_fully_sourced_memo(self):
        """All claims sourced → faithfulness = 1.0."""
        sections = [
            _make_section(
                claims=[
                    _make_claim("Revenue grew", ["s1"]),
                    _make_claim("Market expanding", ["s2"]),
                ]
            )
        ]
        memo = _make_memo(sections=sections)
        result = score_faithfulness(memo)
        assert result.overall_faithfulness == 1.0
        assert result.sourced_claims == 2
        assert result.total_claims == 2
        assert result.orphaned_claims == []

    def test_no_sourced_claims(self):
        """No sourced claims → faithfulness = 0.0."""
        sections = [
            _make_section(
                claims=[
                    _make_claim("Claim A", []),
                    _make_claim("Claim B", []),
                ]
            )
        ]
        memo = _make_memo(sections=sections)
        result = score_faithfulness(memo)
        assert result.overall_faithfulness == 0.0
        assert result.sourced_claims == 0
        assert len(result.orphaned_claims) == 2

    def test_partial_faithfulness(self):
        """2 of 4 claims sourced → faithfulness = 0.5."""
        sections = [
            _make_section(
                claims=[
                    _make_claim("Sourced A", ["s1"]),
                    _make_claim("Sourced B", ["s2"]),
                    _make_claim("Orphan A", []),
                    _make_claim("Orphan B", []),
                ]
            )
        ]
        memo = _make_memo(sections=sections)
        result = score_faithfulness(memo)
        assert result.overall_faithfulness == 0.5
        assert result.sourced_claims == 2
        assert len(result.orphaned_claims) == 2

    def test_empty_memo(self):
        """Memo with no claims → faithfulness = 0.0 (no data)."""
        result = score_faithfulness(_make_memo())
        assert result.total_claims == 0
        assert result.sourced_claims == 0
        assert result.overall_faithfulness == 0.0

    def test_grade_labels(self):
        for score, expected_prefix in [
            (0.95, "A"),
            (0.80, "B"),
            (0.65, "C"),
            (0.45, "D"),
            (0.25, "F"),
        ]:
            sections = [_make_section(claims=[_make_claim("c", ["s1"])])]
            memo = _make_memo(sections=sections)
            # Fake the score via a direct result object
            result = FaithfulnessResult(
                company_name="Co",
                total_claims=100,
                sourced_claims=int(score * 100),
                orphaned_claims=[],
                overall_faithfulness=score,
                per_section=[],
                unique_sources_cited=5,
                total_sources_available=10,
            )
            assert result.grade.startswith(expected_prefix)

    def test_per_section_breakdown(self):
        """Per-section faithfulness should be computed correctly."""
        s1 = _make_section("Financials", claims=[_make_claim("A", ["s1"]), _make_claim("B", [])])
        s2 = _make_section("Risks", claims=[_make_claim("C", ["s2"]), _make_claim("D", ["s3"])])
        memo = _make_memo(sections=[s1, s2])
        result = score_faithfulness(memo)

        assert result.per_section[0].section_title == "Financials"
        assert result.per_section[0].faithfulness_rate == 0.5
        assert result.per_section[1].section_title == "Risks"
        assert result.per_section[1].faithfulness_rate == 1.0

    def test_report_text_renders(self):
        sections = [_make_section(claims=[_make_claim("R", ["s1"])])]
        memo = _make_memo(sections=sections)
        result = score_faithfulness(memo)
        text = faithfulness_report_text(result)
        assert "Faithfulness Report" in text
        assert "Per-Section" in text


# ─── Metrics / coverage tests ─────────────────────────────────────────────────

class TestMetrics:
    def test_memo_text_includes_all_content(self):
        sections = [
            _make_section("Financial Health", claims=[_make_claim("EBITDA margins strong", [])]),
        ]
        memo = _make_memo(sections=sections)
        text = _memo_text(memo)
        assert "financial health" in text.lower()
        assert "ebitda" in text.lower()
        assert "strong fundamentals" in text.lower()  # from executive_summary

    def test_check_coverage_all_found(self):
        text = "apple iphone revenue growing services expanding internationally"
        facts = ["iPhone revenue growing", "services expanding"]
        found, missed = _check_coverage(text, facts)
        assert len(found) == 2
        assert len(missed) == 0

    def test_check_coverage_all_missed(self):
        text = "completely unrelated text about weather and cooking"
        facts = ["iPhone revenue", "cloud services"]
        found, missed = _check_coverage(text, facts)
        assert len(missed) > 0

    def test_check_coverage_empty_list(self):
        found, missed = _check_coverage("anything", [])
        assert found == []
        assert missed == []

    def test_compute_metrics_basic(self):
        from src.evaluation.benchmark import BenchmarkCompany
        benchmark = BenchmarkCompany(
            company_name="TestCorp",
            sector="Technology",
            description="A tech company.",
            known_facts=["revenue growing", "cloud services expanding"],
            known_risks=["competition increasing", "regulatory scrutiny"],
            known_strengths=["strong brand", "loyal customers"],
            expected_sentiment="positive",
            min_expected_confidence=0.5,
        )
        sections = [
            _make_section(
                "Financials",
                claims=[
                    _make_claim("Revenue growing strongly in all markets", ["s1"]),
                    _make_claim("Cloud services expanding rapidly", ["s2"]),
                ],
            ),
            _make_section(
                "Risks",
                claims=[
                    _make_claim("Competition increasing from new entrants", ["s3"]),
                ],
            ),
        ]
        memo = _make_memo(sections=sections, metadata={"total_findings": 3, "total_sources": 3})
        result = compute_metrics(memo, benchmark)

        assert isinstance(result, EvalResult)
        assert result.fact_coverage > 0
        assert result.risk_coverage > 0
        assert 0.0 <= result.composite_score <= 1.0
        assert result.grade in ("A", "B", "C", "D", "F")

    def test_eval_result_serialization(self):
        from src.evaluation.benchmark import BenchmarkCompany
        benchmark = BenchmarkCompany(
            company_name="Co",
            sector="Tech",
            description="A company.",
            known_facts=["fact one"],
            known_risks=["risk one"],
            known_strengths=["strength one"],
            expected_sentiment="neutral",
        )
        sections = [_make_section(claims=[_make_claim("fact one mention here", ["s1"])])]
        memo = _make_memo(sections=sections)
        result = compute_metrics(memo, benchmark)
        d = result.to_dict()
        assert "composite_score" in d
        assert "grade" in d
        assert "faithfulness_rate" in d
        assert isinstance(d["fact_coverage"], float)


# ─── SynthesisAgent._build_memo tests ────────────────────────────────────────

class TestSynthesisBuildMemo:
    """Test SynthesisAgent._build_memo() without making LLM calls."""

    def _get_synthesis_agent(self) -> "SynthesisAgent":
        with patch("src.agents.synthesis.OpenAI"):
            from src.agents.synthesis import SynthesisAgent
            return SynthesisAgent("TestCo")

    def _make_sub_report(self, agent_name: str) -> AgentSubReport:
        return AgentSubReport(
            agent_name=agent_name,
            findings=[AgentClaim(text=f"{agent_name} finding", source_ids=["s1"], confidence=0.8)],
            sources=[_make_source("s1")],
            confidence_score=0.75,
            conflicts=[],
            raw_data_summary="Data summary.",
        )

    def test_build_memo_basic(self):
        agent = self._get_synthesis_agent()
        sub_reports = [
            self._make_sub_report("financial_analyst"),
            self._make_sub_report("team_culture"),
        ]
        tool_input = {
            "executive_summary": "TestCo is a solid investment.",
            "sections": [
                {
                    "title": "Financials",
                    "content": "Strong revenue growth.",
                    "key_claims": [{"text": "Revenue grew 20%", "source_ids": ["s1"], "confidence": 0.85}],
                    "confidence_score": 0.80,
                    "cross_agent_conflicts": [],
                }
            ],
            "overall_confidence": 0.78,
            "investment_highlights": ["Strong growth"],
            "investment_risks": ["Some regulation"],
        }
        memo = agent._build_memo(tool_input, sub_reports, {})
        assert memo.company_name == "TestCo"
        assert memo.executive_summary == "TestCo is a solid investment."
        assert len(memo.sections) == 1
        assert memo.overall_confidence == 0.78
        assert memo.metadata["investment_highlights"] == ["Strong growth"]

    def test_build_memo_confidence_clamping(self):
        agent = self._get_synthesis_agent()
        tool_input = {
            "executive_summary": "Summary.",
            "sections": [{
                "title": "Sec",
                "content": "...",
                "key_claims": [],
                "confidence_score": 1.5,  # Out of bounds
                "cross_agent_conflicts": [],
            }],
            "overall_confidence": -0.2,  # Out of bounds
        }
        memo = agent._build_memo(tool_input, [], {})
        assert 0.0 <= memo.overall_confidence <= 1.0
        assert 0.0 <= memo.sections[0].confidence_score <= 1.0

    def test_build_memo_source_namespacing(self):
        """Source IDs from different agents should both be accessible."""
        agent = self._get_synthesis_agent()
        sub_reports = [
            AgentSubReport(
                agent_name="financial_analyst",
                findings=[],
                sources=[_make_source("s1")],
                confidence_score=0.7,
                conflicts=[],
                raw_data_summary="",
            ),
            AgentSubReport(
                agent_name="risk_sentiment",
                findings=[],
                sources=[_make_source("s1")],  # Same bare ID, different agent
                confidence_score=0.7,
                conflicts=[],
                raw_data_summary="",
            ),
        ]
        tool_input = {
            "executive_summary": "OK",
            "sections": [{
                "title": "T",
                "content": "...",
                "key_claims": [{"text": "A claim", "source_ids": ["s1"], "confidence": 0.8}],
                "confidence_score": 0.7,
                "cross_agent_conflicts": [],
            }],
            "overall_confidence": 0.7,
        }
        # Should not raise; s1 resolves via bare ID fallback
        memo = agent._build_memo(tool_input, sub_reports, {})
        assert len(memo.sections[0].claims[0].source_ids) == 1

    def test_extract_json_from_code_block(self):
        """_extract_json_from_text should handle ```json ... ``` blocks."""
        agent = self._get_synthesis_agent()
        text = '```json\n{"key": "value", "num": 42}\n```'
        result = agent._extract_json_from_text(text)
        assert result == {"key": "value", "num": 42}

    def test_extract_json_from_raw_text(self):
        """_extract_json_from_text should extract bare JSON from text."""
        agent = self._get_synthesis_agent()
        text = 'Here is the output: {"executive_summary": "Great company", "overall_confidence": 0.8}'
        result = agent._extract_json_from_text(text)
        assert result is not None
        assert result["executive_summary"] == "Great company"

    def test_extract_json_invalid(self):
        """_extract_json_from_text should return None for unparseable text."""
        agent = self._get_synthesis_agent()
        result = agent._extract_json_from_text("No JSON here at all!")
        assert result is None
