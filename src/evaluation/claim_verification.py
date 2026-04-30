"""Claim verification heuristics for synthesized investment memos.

This module complements faithfulness scoring. Faithfulness answers whether a
claim cites a source; verification checks whether the cited source registry
contains evidence that appears to support the claim.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.models.schemas import InvestmentMemo

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9&.-]*")
_NUMBER_RE = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?%?(?![\w.])")

_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "amid",
    "among",
    "and",
    "are",
    "because",
    "been",
    "being",
    "between",
    "both",
    "but",
    "can",
    "could",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "into",
    "its",
    "more",
    "most",
    "not",
    "over",
    "per",
    "than",
    "that",
    "the",
    "their",
    "this",
    "through",
    "under",
    "was",
    "were",
    "while",
    "with",
    "would",
    "year",
}


@dataclass
class ClaimVerification:
    """Verification result for a single synthesized claim."""

    section_title: str
    claim_text: str
    source_ids: list[str]
    status: str
    support_score: float
    matched_terms: list[str] = field(default_factory=list)
    missing_numbers: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class VerificationResult:
    """Overall hallucination-risk assessment for a memo."""

    company_name: str
    total_claims: int
    supported_claims: int
    weak_claims: int
    unsupported_claims: int
    missing_source_claims: int
    unresolved_source_claims: int
    overall_score: float
    hallucination_risk: float
    per_claim: list[ClaimVerification]

    @property
    def grade(self) -> str:
        """Human-readable verification grade."""
        if self.overall_score >= 0.90:
            return "A (Strong evidence)"
        if self.overall_score >= 0.75:
            return "B (Mostly supported)"
        if self.overall_score >= 0.60:
            return "C (Mixed support)"
        if self.overall_score >= 0.40:
            return "D (High review need)"
        return "F (Likely hallucination risk)"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the verification result for memo metadata and API clients."""
        data = asdict(self)
        data["grade"] = self.grade
        return data


def verify_memo_claims(memo: InvestmentMemo) -> VerificationResult:
    """Verify synthesized claims against the memo's embedded source registry.

    The scorer is deliberately conservative and deterministic. It uses the
    source title and snippet stored in ``memo.metadata["source_registry"]`` as
    the evidence surface, then flags claims with missing citations, unresolved
    citations, weak keyword overlap, or numeric values absent from the evidence.
    """
    source_registry = memo.metadata.get("source_registry", {}) or {}
    per_claim: list[ClaimVerification] = []

    for section in memo.sections:
        for claim in section.claims:
            per_claim.append(
                _verify_claim(
                    section_title=section.title,
                    claim_text=claim.text,
                    source_ids=list(claim.source_ids),
                    source_registry=source_registry,
                )
            )

    total = len(per_claim)
    supported = sum(1 for c in per_claim if c.status == "supported")
    weak = sum(1 for c in per_claim if c.status == "weak")
    unsupported = sum(1 for c in per_claim if c.status == "unsupported")
    missing = sum(1 for c in per_claim if c.status == "missing_source")
    unresolved = sum(1 for c in per_claim if c.status == "unresolved_source")

    weighted_score = supported + (weak * 0.5)
    overall = weighted_score / total if total else 0.0
    result = VerificationResult(
        company_name=memo.company_name,
        total_claims=total,
        supported_claims=supported,
        weak_claims=weak,
        unsupported_claims=unsupported,
        missing_source_claims=missing,
        unresolved_source_claims=unresolved,
        overall_score=round(overall, 4),
        hallucination_risk=round(1.0 - overall, 4) if total else 0.0,
        per_claim=per_claim,
    )

    logger.info(
        "Verification for %s: %.1f%% (%d supported, %d weak, %d flagged)",
        memo.company_name,
        result.overall_score * 100,
        supported,
        weak,
        unsupported + missing + unresolved,
    )
    return result


def verification_report_text(result: VerificationResult) -> str:
    """Render a verification result as a compact text report."""
    flagged = [
        c
        for c in result.per_claim
        if c.status in {"weak", "unsupported", "missing_source", "unresolved_source"}
    ]
    lines = [
        f"# Claim Verification Report - {result.company_name}",
        f"Overall: {result.overall_score:.1%} [{result.grade}]",
        f"Hallucination risk: {result.hallucination_risk:.1%}",
        (
            "Claims: "
            f"{result.supported_claims} supported, {result.weak_claims} weak, "
            f"{result.unsupported_claims + result.missing_source_claims + result.unresolved_source_claims} flagged"
        ),
    ]
    if flagged:
        lines.extend(["", "## Claims Needing Review"])
        for item in flagged[:10]:
            lines.append(f"- [{item.status}] {item.claim_text[:140]} - {item.reason}")
        if len(flagged) > 10:
            lines.append(f"... and {len(flagged) - 10} more")
    return "\n".join(lines)


def _verify_claim(
    section_title: str,
    claim_text: str,
    source_ids: list[str],
    source_registry: dict[str, Any],
) -> ClaimVerification:
    if not source_ids:
        return ClaimVerification(
            section_title=section_title,
            claim_text=claim_text,
            source_ids=[],
            status="missing_source",
            support_score=0.0,
            reason="Claim has no source IDs attached.",
        )

    resolved_sources = [source_registry[sid] for sid in source_ids if sid in source_registry]
    if not resolved_sources:
        return ClaimVerification(
            section_title=section_title,
            claim_text=claim_text,
            source_ids=source_ids,
            status="unresolved_source",
            support_score=0.0,
            reason="Claim cites source IDs that are absent from the source registry.",
        )

    evidence_text = " ".join(_source_text(src) for src in resolved_sources)
    claim_terms = _meaningful_terms(claim_text)
    evidence_terms = _meaningful_terms(evidence_text)
    matched_terms = sorted(claim_terms & evidence_terms)

    claim_numbers = _numbers(claim_text)
    evidence_numbers = _numbers(evidence_text)
    missing_numbers = sorted(claim_numbers - evidence_numbers)

    denominator = max(1, min(len(claim_terms), 10))
    score = len(matched_terms) / denominator
    if missing_numbers:
        score = min(score, 0.49)

    if score >= 0.45 and not missing_numbers:
        status = "supported"
        reason = "Cited source snippets overlap with the claim terms."
    elif score >= 0.20:
        status = "weak"
        if missing_numbers:
            reason = "Cited evidence overlaps partially, but numeric values are missing."
        else:
            reason = "Cited evidence has only partial overlap with the claim."
    else:
        status = "unsupported"
        reason = "Cited evidence has little overlap with the claim."

    return ClaimVerification(
        section_title=section_title,
        claim_text=claim_text,
        source_ids=source_ids,
        status=status,
        support_score=round(score, 4),
        matched_terms=matched_terms[:12],
        missing_numbers=missing_numbers,
        reason=reason,
    )


def _source_text(source: Any) -> str:
    if isinstance(source, dict):
        fields = [
            source.get("title", ""),
            source.get("snippet", ""),
            source.get("source_type", ""),
            source.get("originating_agent", ""),
        ]
        return " ".join(str(field) for field in fields if field)
    fields = [
        getattr(source, "title", ""),
        getattr(source, "snippet", ""),
        getattr(source, "source_type", ""),
    ]
    return " ".join(str(field) for field in fields if field)


def _meaningful_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in _TOKEN_RE.findall(text.lower()):
        token = raw.strip(".-")
        if not token or token in _STOPWORDS:
            continue
        if token.isdigit() or len(token) >= 4:
            terms.add(_normalize_number(token) if any(ch.isdigit() for ch in token) else token)
    return terms


def _numbers(text: str) -> set[str]:
    return {_normalize_number(match.group(0).lower()) for match in _NUMBER_RE.finditer(text)}


def _normalize_number(token: str) -> str:
    return token.replace(",", "")
