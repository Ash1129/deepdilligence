"""Faithfulness scoring: measures claim-to-source traceability in an InvestmentMemo."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.models.schemas import AgentClaim, InvestmentMemo, SynthesizedSection

logger = logging.getLogger(__name__)


# ─── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class SectionFaithfulness:
    """Faithfulness breakdown for a single memo section."""

    section_title: str
    total_claims: int
    sourced_claims: int          # claims with ≥1 source_id
    orphaned_claims: list[str]   # claim texts with no source
    faithfulness_rate: float     # sourced / total (0.0 if total == 0)


@dataclass
class FaithfulnessResult:
    """Overall faithfulness score for a complete InvestmentMemo."""

    company_name: str
    total_claims: int
    sourced_claims: int
    orphaned_claims: list[str]           # all orphaned claim texts across memo
    overall_faithfulness: float          # 0.0–1.0
    per_section: list[SectionFaithfulness]

    # Source diversity
    unique_sources_cited: int            # distinct source IDs referenced by claims
    total_sources_available: int         # total sources in memo metadata

    # Convenience label
    @property
    def grade(self) -> str:
        """Human-readable faithfulness grade."""
        if self.overall_faithfulness >= 0.90:
            return "A (Excellent)"
        if self.overall_faithfulness >= 0.75:
            return "B (Good)"
        if self.overall_faithfulness >= 0.60:
            return "C (Acceptable)"
        if self.overall_faithfulness >= 0.40:
            return "D (Needs improvement)"
        return "F (Poor)"


# ─── Scorer ──────────────────────────────────────────────────────────────────

def score_faithfulness(memo: InvestmentMemo) -> FaithfulnessResult:
    """Compute faithfulness (source traceability) scores for a memo.

    A claim is "faithful" if it references at least one source_id.
    Faithfulness rate = sourced_claims / total_claims.

    Args:
        memo: The InvestmentMemo to evaluate.

    Returns:
        FaithfulnessResult with per-section breakdown and overall score.
    """
    all_orphaned: list[str] = []
    cited_source_ids: set[str] = set()
    per_section: list[SectionFaithfulness] = []

    total_claims = 0
    total_sourced = 0

    for section in memo.sections:
        s_total = len(section.claims)
        s_sourced = 0
        s_orphaned: list[str] = []

        for claim in section.claims:
            if claim.source_ids:
                s_sourced += 1
                cited_source_ids.update(claim.source_ids)
            else:
                s_orphaned.append(claim.text)
                all_orphaned.append(claim.text)

        rate = s_sourced / s_total if s_total > 0 else 0.0

        per_section.append(
            SectionFaithfulness(
                section_title=section.title,
                total_claims=s_total,
                sourced_claims=s_sourced,
                orphaned_claims=s_orphaned,
                faithfulness_rate=rate,
            )
        )

        total_claims += s_total
        total_sourced += s_sourced

    overall = total_sourced / total_claims if total_claims > 0 else 0.0

    # Total sources from metadata (summed across agents)
    total_available = memo.metadata.get("total_sources", 0)

    result = FaithfulnessResult(
        company_name=memo.company_name,
        total_claims=total_claims,
        sourced_claims=total_sourced,
        orphaned_claims=all_orphaned,
        overall_faithfulness=overall,
        per_section=per_section,
        unique_sources_cited=len(cited_source_ids),
        total_sources_available=total_available,
    )

    logger.info(
        "Faithfulness for %s: %.1f%% (%d/%d claims sourced) — grade: %s",
        memo.company_name,
        overall * 100,
        total_sourced,
        total_claims,
        result.grade,
    )
    return result


def faithfulness_report_text(result: FaithfulnessResult) -> str:
    """Render a faithfulness result as a human-readable text block."""
    lines = [
        f"# Faithfulness Report — {result.company_name}",
        f"Overall: {result.overall_faithfulness:.1%}  [{result.grade}]",
        f"Claims sourced: {result.sourced_claims} / {result.total_claims}",
        f"Unique source IDs cited: {result.unique_sources_cited} / {result.total_sources_available}",
        "",
        "## Per-Section Breakdown",
    ]
    for s in result.per_section:
        lines.append(
            f"  {s.section_title}: {s.faithfulness_rate:.1%} "
            f"({s.sourced_claims}/{s.total_claims})"
        )
        for orphan in s.orphaned_claims[:3]:
            lines.append(f"    ⚠ Unsourced: \"{orphan[:100]}\"")

    if result.orphaned_claims:
        lines += ["", f"## Orphaned Claims ({len(result.orphaned_claims)} total)"]
        for c in result.orphaned_claims[:10]:
            lines.append(f"  - {c[:120]}")
        if len(result.orphaned_claims) > 10:
            lines.append(f"  ... and {len(result.orphaned_claims) - 10} more")

    return "\n".join(lines)
