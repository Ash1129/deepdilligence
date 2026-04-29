"""Evaluation metrics: coverage, faithfulness, confidence calibration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.evaluation.benchmark import BenchmarkCompany
from src.evaluation.faithfulness import FaithfulnessResult, score_faithfulness
from src.models.schemas import InvestmentMemo
from src.utils.config import BENCHMARKS_DIR

logger = logging.getLogger(__name__)


# ─── Result model ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Complete evaluation result for one company memo."""

    company_name: str
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    # Coverage — how much ground truth does the memo capture?
    fact_coverage: float = 0.0          # 0.0–1.0: % of known_facts mentioned
    risk_coverage: float = 0.0          # 0.0–1.0: % of known_risks flagged
    strength_coverage: float = 0.0      # 0.0–1.0: % of known_strengths recognized

    facts_found: list[str] = field(default_factory=list)
    facts_missed: list[str] = field(default_factory=list)
    risks_found: list[str] = field(default_factory=list)
    risks_missed: list[str] = field(default_factory=list)
    strengths_found: list[str] = field(default_factory=list)
    strengths_missed: list[str] = field(default_factory=list)

    # Faithfulness
    faithfulness: FaithfulnessResult | None = None

    # Memo quality
    section_count: int = 0
    total_findings: int = 0
    total_sources: int = 0
    overall_confidence: float = 0.0
    conflict_count: int = 0

    # Sentiment calibration
    expected_sentiment: str = ""
    confidence_vs_floor: float = 0.0   # overall_confidence - min_expected_confidence

    @property
    def composite_score(self) -> float:
        """Weighted composite: 40% coverage, 40% faithfulness, 20% confidence."""
        coverage_avg = (self.fact_coverage + self.risk_coverage + self.strength_coverage) / 3
        faithfulness_score = self.faithfulness.overall_faithfulness if self.faithfulness else 0.0
        return 0.40 * coverage_avg + 0.40 * faithfulness_score + 0.20 * self.overall_confidence

    @property
    def grade(self) -> str:
        score = self.composite_score
        if score >= 0.85:
            return "A"
        if score >= 0.70:
            return "B"
        if score >= 0.55:
            return "C"
        if score >= 0.40:
            return "D"
        return "F"

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON export)."""
        return {
            "company_name": self.company_name,
            "evaluated_at": self.evaluated_at.isoformat(),
            "fact_coverage": round(self.fact_coverage, 4),
            "risk_coverage": round(self.risk_coverage, 4),
            "strength_coverage": round(self.strength_coverage, 4),
            "facts_found": self.facts_found,
            "facts_missed": self.facts_missed,
            "risks_found": self.risks_found,
            "risks_missed": self.risks_missed,
            "strengths_found": self.strengths_found,
            "strengths_missed": self.strengths_missed,
            "faithfulness_rate": round(
                self.faithfulness.overall_faithfulness if self.faithfulness else 0.0, 4
            ),
            "faithfulness_grade": self.faithfulness.grade if self.faithfulness else "N/A",
            "section_count": self.section_count,
            "total_findings": self.total_findings,
            "total_sources": self.total_sources,
            "overall_confidence": round(self.overall_confidence, 4),
            "conflict_count": self.conflict_count,
            "expected_sentiment": self.expected_sentiment,
            "composite_score": round(self.composite_score, 4),
            "grade": self.grade,
        }


# ─── Keyword coverage checker ─────────────────────────────────────────────────

def _memo_text(memo: InvestmentMemo) -> str:
    """Flatten a memo into a single lowercase searchable string."""
    parts = [memo.executive_summary]
    for section in memo.sections:
        parts.append(section.title)
        parts.append(section.content)
        for claim in section.claims:
            parts.append(claim.text)
    return " ".join(parts).lower()


def _check_coverage(memo_text: str, items: list[str]) -> tuple[list[str], list[str]]:
    """Check which ground-truth items appear (as keywords) in the memo.

    A ground-truth item is considered "found" if ANY of its words (>4 chars)
    appear in the memo text. This is a lenient fuzzy match — good enough for
    course-level evaluation without requiring a full semantic similarity model.

    Returns:
        Tuple (found_items, missed_items).
    """
    found: list[str] = []
    missed: list[str] = []

    for item in items:
        # Use significant words (length > 4) as keywords
        keywords = [w.lower() for w in item.split() if len(w) > 4]
        if not keywords:
            keywords = [item.lower()[:20]]

        # Item is "found" if at least half of its keywords appear
        matches = sum(1 for kw in keywords if kw in memo_text)
        threshold = max(1, len(keywords) // 2)

        if matches >= threshold:
            found.append(item)
        else:
            missed.append(item)

    return found, missed


# ─── Main evaluator ──────────────────────────────────────────────────────────

def compute_metrics(memo: InvestmentMemo, benchmark: BenchmarkCompany) -> EvalResult:
    """Evaluate a memo against a benchmark ground-truth profile.

    Computes:
    - Fact coverage: % of known_facts mentioned in memo
    - Risk coverage: % of known_risks flagged
    - Strength coverage: % of known_strengths recognized
    - Faithfulness: source traceability via score_faithfulness()
    - Composite score and letter grade

    Args:
        memo: The InvestmentMemo produced by the pipeline.
        benchmark: BenchmarkCompany with ground-truth facts, risks, strengths.

    Returns:
        EvalResult with all metrics populated.
    """
    text = _memo_text(memo)

    facts_found, facts_missed = _check_coverage(text, benchmark.known_facts)
    risks_found, risks_missed = _check_coverage(text, benchmark.known_risks)
    strengths_found, strengths_missed = _check_coverage(text, benchmark.known_strengths)

    fact_coverage = len(facts_found) / len(benchmark.known_facts) if benchmark.known_facts else 1.0
    risk_coverage = len(risks_found) / len(benchmark.known_risks) if benchmark.known_risks else 1.0
    strength_coverage = len(strengths_found) / len(benchmark.known_strengths) if benchmark.known_strengths else 1.0

    faithfulness = score_faithfulness(memo)

    # Count cross-agent conflicts across all sections
    conflict_count = sum(len(s.conflicting_claims) for s in memo.sections)

    result = EvalResult(
        company_name=memo.company_name,
        fact_coverage=fact_coverage,
        risk_coverage=risk_coverage,
        strength_coverage=strength_coverage,
        facts_found=facts_found,
        facts_missed=facts_missed,
        risks_found=risks_found,
        risks_missed=risks_missed,
        strengths_found=strengths_found,
        strengths_missed=strengths_missed,
        faithfulness=faithfulness,
        section_count=len(memo.sections),
        total_findings=memo.metadata.get("total_findings", 0),
        total_sources=memo.metadata.get("total_sources", 0),
        overall_confidence=memo.overall_confidence,
        conflict_count=conflict_count,
        expected_sentiment=benchmark.expected_sentiment,
        confidence_vs_floor=memo.overall_confidence - benchmark.min_expected_confidence,
    )

    logger.info(
        "Eval %s: composite=%.2f (%s) | facts=%.0f%% risks=%.0f%% faithful=%.0f%%",
        memo.company_name,
        result.composite_score,
        result.grade,
        fact_coverage * 100,
        risk_coverage * 100,
        faithfulness.overall_faithfulness * 100,
    )
    return result


def save_eval_result(result: EvalResult, output_dir: Path | None = None) -> Path:
    """Save an EvalResult to JSON in the benchmarks directory.

    Args:
        result: Populated EvalResult.
        output_dir: Directory to write to (default: BENCHMARKS_DIR/results/).

    Returns:
        Path to the written JSON file.
    """
    out_dir = output_dir or (BENCHMARKS_DIR / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = result.company_name.lower().replace(" ", "_").replace("/", "_")
    path = out_dir / f"eval_{safe_name}_{result.evaluated_at.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Saved eval result → %s", path)
    return path


def metrics_report_text(result: EvalResult) -> str:
    """Render an EvalResult as a human-readable text report."""
    faith = result.faithfulness
    lines = [
        f"# Evaluation Report — {result.company_name}",
        f"Composite Score: {result.composite_score:.1%}  [Grade: {result.grade}]",
        f"Expected sentiment: {result.expected_sentiment}",
        "",
        "## Coverage",
        f"  Facts:     {result.fact_coverage:.1%} ({len(result.facts_found)}/{len(result.facts_found)+len(result.facts_missed)} known facts found)",
        f"  Risks:     {result.risk_coverage:.1%} ({len(result.risks_found)}/{len(result.risks_found)+len(result.risks_missed)} known risks found)",
        f"  Strengths: {result.strength_coverage:.1%} ({len(result.strengths_found)}/{len(result.strengths_found)+len(result.strengths_missed)} known strengths found)",
    ]

    if result.facts_missed:
        lines.append("  Missed facts: " + "; ".join(result.facts_missed[:3]))
    if result.risks_missed:
        lines.append("  Missed risks: " + "; ".join(result.risks_missed[:3]))

    if faith:
        lines += [
            "",
            "## Faithfulness",
            f"  Overall: {faith.overall_faithfulness:.1%}  [{faith.grade}]",
            f"  Sourced claims: {faith.sourced_claims} / {faith.total_claims}",
            f"  Unique sources cited: {faith.unique_sources_cited} / {faith.total_sources_available}",
        ]

    lines += [
        "",
        "## Memo Quality",
        f"  Sections: {result.section_count}",
        f"  Total findings: {result.total_findings}",
        f"  Total sources: {result.total_sources}",
        f"  Overall confidence: {result.overall_confidence:.1%}",
        f"  Cross-agent conflicts surfaced: {result.conflict_count}",
    ]

    return "\n".join(lines)
