"""Synthesis agent — reconciles specialist findings into a final InvestmentMemo."""

import json
import logging
from datetime import datetime
from typing import Any

import json
import re

from openai import OpenAI

from src.models.prompts import PRODUCE_MEMO_TOOL, SYNTHESIS_SYSTEM
from src.models.schemas import (
    AgentClaim,
    AgentSubReport,
    ConflictingClaim,
    InvestmentMemo,
    Source,
    SynthesizedSection,
)
from src.utils.config import SYNTHESIS_MODEL, get_openai_api_key

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """Reconciles the four specialist sub-reports into a unified InvestmentMemo.

    Architecture note: this agent deliberately never sees raw data — only the
    structured AgentSubReport objects produced by the specialist agents.
    It operates at the level of claims and evidence, not source documents.

    Responsibilities:
    - Deduplicate overlapping claims across agents
    - Detect and explicitly surface cross-agent contradictions
    - Assign section-level confidence scores
    - Produce a balanced executive summary and Investment Thesis section
    """

    def __init__(self, company_name: str) -> None:
        self.company_name = company_name
        self._client = OpenAI(api_key=get_openai_api_key())
        self.logger = logging.getLogger(f"{__name__}.synthesis")

    # ─── Public API ──────────────────────────────────────────────────────────

    def synthesize(
        self,
        sub_reports: list[AgentSubReport],
        extra_metadata: dict | None = None,
    ) -> InvestmentMemo:
        """Synthesize specialist sub-reports into a final InvestmentMemo.

        Args:
            sub_reports: List of AgentSubReport from the four specialist agents.
            extra_metadata: Optional dict merged into memo.metadata (e.g. timing info).

        Returns:
            Fully populated InvestmentMemo with sections, conflicts, and confidence.
        """
        self.logger.info(
            "Synthesizing %d sub-reports for %s", len(sub_reports), self.company_name
        )

        formatted = self._format_sub_reports(sub_reports)

        user_message = (
            f"Company: {self.company_name}\n\n"
            "=== SPECIALIST AGENT REPORTS ===\n"
            f"{formatted}\n\n"
            "Synthesize these reports into a unified investment memo.\n"
            "Pay special attention to cross-agent conflicts — flag every one explicitly.\n"
            "Call produce_investment_memo with your complete analysis."
        )

        response = self._client.chat.completions.create(
            model=SYNTHESIS_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            tools=[PRODUCE_MEMO_TOOL],
            tool_choice={"type": "function", "function": {"name": "produce_investment_memo"}},
            max_completion_tokens=32000,
        )

        message = response.choices[0].message

        # ── Primary path: forced tool call returned as expected ──
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "produce_investment_memo":
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        self.logger.error("Failed to parse memo arguments: %s", e)
                        break
                    memo = self._build_memo(
                        tool_input=tool_input,
                        sub_reports=sub_reports,
                        extra_metadata=extra_metadata or {},
                    )
                    self.logger.info(
                        "Synthesis complete — %d sections, overall_confidence=%.2f",
                        len(memo.sections),
                        memo.overall_confidence,
                    )
                    return memo

        # ── Fallback: model returned text instead of tool call ──
        # Some models ignore tool_choice and write JSON as text; try to extract it.
        self.logger.warning(
            "produce_investment_memo tool not called (finish_reason=%s). "
            "Attempting JSON extraction from text response.",
            response.choices[0].finish_reason,
        )
        if message.content:
            self.logger.debug("Raw synthesis text (first 500 chars): %s", message.content[:500])
            extracted = self._extract_json_from_text(message.content)
            if extracted:
                self.logger.info("Successfully extracted JSON from text fallback")
                return self._build_memo(extracted, sub_reports, extra_metadata or {})

        self.logger.error("Synthesis completely failed — returning empty memo")
        return InvestmentMemo(
            company_name=self.company_name,
            executive_summary="Synthesis failed: the model did not return a structured memo.",
            sections=[],
            overall_confidence=0.0,
            metadata={"error": "tool call not returned"},
        )

    # ─── JSON fallback extraction ─────────────────────────────────────────────

    def _extract_json_from_text(self, text: str) -> dict | None:
        """Try to extract a JSON object from a text response.

        Handles cases where the model writes ```json ... ``` blocks or raw JSON
        instead of using the tool call as instructed.
        """
        # Try ```json ... ``` code block first
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding the outermost { ... } in the text
        start = text.find("{")
        if start != -1:
            # Walk forward to find the matching closing brace
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    # ─── Formatting ──────────────────────────────────────────────────────────

    def _format_sub_reports(self, sub_reports: list[AgentSubReport]) -> str:
        """Render all sub-reports as structured text for the synthesis prompt.

        Includes agent name, all findings with confidence, all sources,
        internal conflicts, and the raw data summary.
        """
        parts: list[str] = []

        for report in sub_reports:
            parts.append(f"{'=' * 60}")
            parts.append(f"AGENT: {report.agent_name.upper()}")
            parts.append(f"Overall confidence: {report.confidence_score:.2f}")
            parts.append(f"Raw data coverage: {report.raw_data_summary}")
            parts.append("")

            # Sources registry
            if report.sources:
                parts.append(f"SOURCES ({len(report.sources)}):")
                for src in report.sources:
                    parts.append(f"  [{src.id}] {src.title} ({src.source_type}) — {src.url}")
                    if src.snippet:
                        parts.append(f"         Snippet: {src.snippet[:120]}")
                parts.append("")

            # Findings
            if report.findings:
                parts.append(f"FINDINGS ({len(report.findings)}):")
                for i, claim in enumerate(report.findings, 1):
                    source_refs = ", ".join(claim.source_ids) if claim.source_ids else "no source"
                    parts.append(
                        f"  {i}. [{claim.confidence:.2f}] {claim.text}"
                        f"  (sources: {source_refs})"
                    )
                parts.append("")
            else:
                parts.append("FINDINGS: None (agent returned no findings)\n")

            # Internal conflicts
            if report.conflicts:
                parts.append(f"INTERNAL CONFLICTS ({len(report.conflicts)}):")
                for conflict in report.conflicts:
                    parts.append(f"  ⚠ {conflict.description}")
                    parts.append(f"    A: {conflict.claim_a.text}")
                    parts.append(f"    B: {conflict.claim_b.text}")
                parts.append("")

        return "\n".join(parts)

    # ─── Memo construction ───────────────────────────────────────────────────

    def _build_memo(
        self,
        tool_input: dict,
        sub_reports: list[AgentSubReport],
        extra_metadata: dict,
    ) -> InvestmentMemo:
        """Convert the produce_investment_memo tool output into a validated InvestmentMemo.

        Preserves source objects from the originating agent reports so that all
        citations in the final memo are traceable to the original scraped data.
        """
        source_lookup, source_registry, bare_source_aliases = self._build_source_registry(sub_reports)

        # Build each SynthesizedSection
        sections: list[SynthesizedSection] = []
        for sec in tool_input.get("sections", []):
            key_claims = self._build_claims(
                sec.get("key_claims", []),
                source_lookup,
                bare_source_aliases,
            )
            cross_conflicts = self._build_cross_conflicts(sec.get("cross_agent_conflicts", []))

            sections.append(
                SynthesizedSection(
                    title=str(sec.get("title", "Untitled")),
                    content=str(sec.get("content", "")),
                    claims=key_claims,
                    confidence_score=float(
                        max(0.0, min(1.0, sec.get("confidence_score", 0.5)))
                    ),
                    conflicting_claims=cross_conflicts,
                )
            )

        # Build metadata
        agent_confidences = {r.agent_name: r.confidence_score for r in sub_reports}

        metadata = {
            "generated_by": "DeepDiligence v1",
            "specialist_confidences": agent_confidences,
            "agent_count": len(sub_reports),
            "total_findings": sum(len(r.findings) for r in sub_reports),
            "total_sources": sum(len(r.sources) for r in sub_reports),
            "investment_highlights": tool_input.get("investment_highlights", []),
            "investment_risks": tool_input.get("investment_risks", []),
            "source_registry": source_registry,
            **extra_metadata,
        }

        return InvestmentMemo(
            company_name=self.company_name,
            generated_at=datetime.utcnow(),
            executive_summary=str(tool_input.get("executive_summary", "")),
            sections=sections,
            overall_confidence=float(
                max(0.0, min(1.0, tool_input.get("overall_confidence", 0.5)))
            ),
            metadata=metadata,
        )

    def _build_source_registry(
        self,
        sub_reports: list[AgentSubReport],
    ) -> tuple[dict[str, Source], dict[str, dict], dict[str, str]]:
        """Build a self-contained, namespaced source registry.

        Returns:
            source_lookup: Namespaced source ID → Source object for validation.
            source_registry: Namespaced source ID → JSON-serializable source metadata.
            bare_source_aliases: Bare source ID → namespaced ID only when unambiguous.
        """
        source_lookup: dict[str, Source] = {}
        source_registry: dict[str, dict] = {}
        bare_to_namespaced: dict[str, list[str]] = {}

        for report in sub_reports:
            for src in report.sources:
                original_id = str(src.id)
                namespaced_id = f"{report.agent_name}::{original_id}"
                source_lookup[namespaced_id] = src
                bare_to_namespaced.setdefault(original_id, []).append(namespaced_id)
                source_registry[namespaced_id] = {
                    "id": namespaced_id,
                    "original_id": original_id,
                    "originating_agent": report.agent_name,
                    "url": src.url,
                    "title": src.title,
                    "snippet": src.snippet,
                    "source_type": (
                        src.source_type.value
                        if hasattr(src.source_type, "value")
                        else str(src.source_type)
                    ),
                    "retrieved_at": src.retrieved_at.isoformat(),
                }

        bare_source_aliases = {
            original_id: ids[0]
            for original_id, ids in bare_to_namespaced.items()
            if len(ids) == 1
        }
        return source_lookup, source_registry, bare_source_aliases

    def _build_claims(
        self,
        raw_claims: list[dict],
        source_lookup: dict[str, Source],
        bare_source_aliases: dict[str, str],
    ) -> list[AgentClaim]:
        """Build AgentClaim objects with source_ids resolved to registry keys."""
        claims: list[AgentClaim] = []
        for c in raw_claims:
            originating_agent = str(c.get("originating_agent", ""))
            valid_ids = []
            for sid in c.get("source_ids", []):
                resolved = self._resolve_source_id(
                    source_id=str(sid),
                    originating_agent=originating_agent,
                    source_lookup=source_lookup,
                    bare_source_aliases=bare_source_aliases,
                )
                if resolved and resolved not in valid_ids:
                    valid_ids.append(resolved)
            claims.append(
                AgentClaim(
                    text=str(c.get("text", "")),
                    source_ids=valid_ids,
                    confidence=float(max(0.0, min(1.0, c.get("confidence", 0.5)))),
                )
            )
        return claims

    @staticmethod
    def _resolve_source_id(
        source_id: str,
        originating_agent: str,
        source_lookup: dict[str, Source],
        bare_source_aliases: dict[str, str],
    ) -> str | None:
        """Resolve model-emitted source IDs to canonical source_registry keys."""
        if source_id in source_lookup:
            return source_id

        if originating_agent:
            candidate = f"{originating_agent}::{source_id}"
            if candidate in source_lookup:
                return candidate

        return bare_source_aliases.get(source_id)

    def _build_cross_conflicts(self, raw_conflicts: list[dict]) -> list[ConflictingClaim]:
        """Build ConflictingClaim objects from cross-agent conflict dicts."""
        conflicts: list[ConflictingClaim] = []
        for c in raw_conflicts:
            agent_a = c.get("agent_a", "unknown")
            agent_b = c.get("agent_b", "unknown")
            conflicts.append(
                ConflictingClaim(
                    claim_a=AgentClaim(
                        text=f"[{agent_a}] {c.get('claim_a', '')}",
                        source_ids=[],
                        confidence=0.5,
                    ),
                    claim_b=AgentClaim(
                        text=f"[{agent_b}] {c.get('claim_b', '')}",
                        source_ids=[],
                        confidence=0.5,
                    ),
                    description=str(c.get("description", "")),
                )
            )
        return conflicts
