"""ReactAgent: intermediate base class providing the ReAct loop and structured output.

Both steps run through the Claude Code CLI (see src/utils/claude_cli.py):

1. Research: the CLI drives the Reason → Act → Observe loop itself. It can call
   this agent's own data tools (served over MCP by src/tools/mcp_server.py) and
   the CLI's built-in web search, so every run works from live data.
2. Analysis: a second, tool-free call turns the gathered evidence into an
   AgentSubReport, constrained to the produce_analysis JSON schema.
"""

import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.models.prompts import LIVE_RESEARCH_RULES, PRODUCE_ANALYSIS_TOOL
from src.models.schemas import (
    AgentClaim,
    AgentSubReport,
    ConflictingClaim,
    Source,
    SourceType,
)
from src.utils.claude_cli import (
    WEB_TOOLS,
    ClaudeCLIError,
    extract_json_object,
    mcp_config_for_agent,
    run_claude,
)
from src.utils.config import (
    AGENT_MODEL,
    CACHE_DIR,
    CLAUDE_ANALYZE_TIMEOUT,
    CLAUDE_GATHER_TIMEOUT,
    cache_enabled,
)

logger = logging.getLogger(__name__)

# Per-source character budget when evidence is rendered for the analysis step.
MAX_SOURCE_CHARS = 8000


class ReactAgent(BaseAgent):
    """Extends BaseAgent with the ReAct pattern and optional disk-cached runs.

    Provides two shared capabilities for specialist agents:
    1. _run_react_loop(): LLM-driven adaptive data gathering via tool use.
    2. _produce_structured_report(): structured AgentSubReport via a JSON schema.

    Specialist agents implement gather_data() and analyze() using these helpers,
    declare their tools in `TOOLS`, and route them in `_execute_tool()`.
    """

    # Function-style tool definitions ({"type": "function", "function": {...}}).
    # Served to the CLI by src/tools/mcp_server.py.
    TOOLS: list[dict] = []

    def __init__(self, company_name: str, ticker: str | None = None, **kwargs: Any) -> None:
        super().__init__(company_name, ticker, **kwargs)

    def _execute_tool(self, name: str, tool_input: dict) -> Any:
        """Route one tool call to a data function. Implemented by each specialist."""
        raise NotImplementedError(f"{type(self).__name__} defines no tools")

    # ─── Cache helpers ────────────────────────────────────────────────────────

    def _cache_key(self) -> str:
        """Deterministic 16-char hex key from (agent_name, company, ticker)."""
        raw = f"{self.agent_name}|{self.company_name}|{self.ticker or ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _agent_cache_dir(self) -> Path:
        """Cache subdirectory for this agent."""
        return CACHE_DIR / "agents" / self.agent_name

    def _report_cache_path(self) -> Path:
        """Disk path for the full cached AgentSubReport."""
        return self._agent_cache_dir() / f"{self._cache_key()}_report.json"

    # ─── ReAct loop ──────────────────────────────────────────────────────────

    def _run_react_loop(
        self,
        system_prompt: str,
        initial_user_message: str,
    ) -> tuple[str, dict[str, Any]]:
        """Run the research loop through the Claude CLI and collect every tool result.

        The CLI performs Reason → Act (tool call) → Observe until the model decides
        it has enough evidence. Each tool call is recorded so analyze() can
        enumerate sources.

        Args:
            system_prompt: Agent's research system prompt.
            initial_user_message: Starting research request.

        Returns:
            Tuple of (model's closing summary, collected_data_dict).

        Raises:
            ClaudeCLIError: If the CLI failed before gathering anything.
        """
        tool_names = [tool["function"]["name"] for tool in self.TOOLS]

        result = run_claude(
            prompt=f"Today's date: {date.today().isoformat()}\n\n{initial_user_message}",
            system_prompt=f"{system_prompt}\n\n{LIVE_RESEARCH_RULES}",
            model=AGENT_MODEL,
            builtin_tools=WEB_TOOLS,
            mcp_tools=tool_names,
            mcp_config=mcp_config_for_agent(self.agent_name) if tool_names else None,
            timeout=CLAUDE_GATHER_TIMEOUT,
            raise_on_error=False,
        )

        collected_data: dict[str, Any] = {}
        for index, event in enumerate(result.tool_events):
            # Unique key per call so analyze() can enumerate sources.
            suffix = (event.get("id") or "")[-8:] or str(index)
            data_key = f"{event['tool']}_{suffix}"
            if data_key in collected_data:
                data_key = f"{data_key}_{index}"
            collected_data[data_key] = {
                "tool": event["tool"],
                "input": event.get("input") or {},
                "output": event.get("output"),
            }

        if result.is_error:
            if not collected_data:
                raise ClaudeCLIError(f"{self.agent_name} research failed: {result.error}")
            # Keep whatever was gathered before the failure, as before.
            self.logger.warning(
                "Research ended early (%s) — continuing with %d sources",
                result.error, len(collected_data),
            )

        self.logger.info(
            "%s: ReAct loop done — %d sources collected in %d turns",
            self.agent_name,
            len(collected_data),
            result.num_turns,
        )
        return result.text, collected_data

    # ─── Structured output ───────────────────────────────────────────────────

    def _produce_structured_report(
        self, system_prompt: str, raw_data: dict[str, Any]
    ) -> AgentSubReport:
        """Turn gathered data into a structured AgentSubReport.

        The reply is constrained to the produce_analysis JSON schema, so the
        model returns data matching the AgentSubReport contract.

        Args:
            system_prompt: Analysis system prompt for this agent.
            raw_data: Output of gather_data().

        Returns:
            Validated AgentSubReport instance.
        """
        data_text = self._format_raw_data(raw_data)

        user_message = (
            f"Company under analysis: {self.company_name}\n"
            f"Ticker: {self.ticker or 'not available (private or unknown)'}\n"
            f"Today's date: {date.today().isoformat()}\n\n"
            "=== GATHERED DATA ===\n"
            f"{data_text}\n\n"
            "Analyze the data above and return your structured findings. "
            "Every finding must reference at least one source ID. Be specific — no vague claims."
        )

        result = run_claude(
            prompt=user_message,
            system_prompt=system_prompt,
            model=AGENT_MODEL,
            json_schema=PRODUCE_ANALYSIS_TOOL["function"]["parameters"],
            timeout=CLAUDE_ANALYZE_TIMEOUT,
            raise_on_error=False,
        )

        payload = result.structured or extract_json_object(result.text)
        if payload:
            return self._build_subreport(payload)

        self.logger.error(
            "Structured analysis not returned (%s) — building empty fallback report",
            result.error or "no JSON in reply",
        )
        return AgentSubReport(
            agent_name=self.agent_name,
            findings=[],
            sources=[],
            confidence_score=0.0,
            conflicts=[],
            raw_data_summary=(
                "Analysis step failed: the model did not return a structured report"
                + (f" ({result.error})." if result.error else ".")
            ),
        )

    def _format_raw_data(self, raw_data: dict[str, Any]) -> str:
        """Render the collected_data dict as readable text for the analysis prompt."""
        parts: list[str] = []
        for key, value in raw_data.items():
            if key.startswith("_"):
                continue

            parts.append(f"--- {key} ---")
            if isinstance(value, dict) and "tool" in value:
                parts.append(f"Tool: {value['tool']}")
                parts.append(f"Input: {json.dumps(value['input'], default=str)}")
                output = value["output"]
                output_str = output if isinstance(output, str) else json.dumps(output, default=str)
                if len(output_str) > MAX_SOURCE_CHARS:
                    output_str = output_str[:MAX_SOURCE_CHARS] + "... [truncated]"
                parts.append(f"Output: {output_str}")
            else:
                raw_str = json.dumps(value, default=str)
                parts.append(raw_str[:MAX_SOURCE_CHARS] if len(raw_str) > MAX_SOURCE_CHARS else raw_str)
            parts.append("")

        return "\n".join(parts) if parts else "No data was gathered."

    def _build_subreport(self, tool_input: dict) -> AgentSubReport:
        """Convert the produce_analysis payload into a validated AgentSubReport."""
        # Build Source registry
        source_map: dict[str, Source] = {}
        for s in tool_input.get("sources", []):
            try:
                stype = SourceType(s.get("source_type", "other"))
            except ValueError:
                stype = SourceType.OTHER

            src = Source(
                id=str(s.get("id", f"src_{len(source_map)}")),
                url=str(s.get("url", "")),
                title=str(s.get("title", "Unknown source")),
                snippet=str(s.get("snippet", ""))[:500],
                source_type=stype,
            )
            source_map[src.id] = src

        # Build AgentClaim list
        findings: list[AgentClaim] = []
        for f in tool_input.get("findings", []):
            valid_ids = [sid for sid in f.get("source_ids", []) if sid in source_map]
            claim = AgentClaim(
                text=str(f.get("text", "")),
                source_ids=valid_ids,
                confidence=float(max(0.0, min(1.0, f.get("confidence", 0.5)))),
            )
            findings.append(claim)

        # Build ConflictingClaim list
        conflicts: list[ConflictingClaim] = []
        for c in tool_input.get("conflicts", []):
            conflict = ConflictingClaim(
                claim_a=AgentClaim(
                    text=str(c.get("claim_a_text", "")), source_ids=[], confidence=0.5
                ),
                claim_b=AgentClaim(
                    text=str(c.get("claim_b_text", "")), source_ids=[], confidence=0.5
                ),
                description=str(c.get("description", "")),
            )
            conflicts.append(conflict)

        return AgentSubReport(
            agent_name=self.agent_name,
            findings=findings,
            sources=list(source_map.values()),
            confidence_score=float(
                max(0.0, min(1.0, tool_input.get("confidence_score", 0.5)))
            ),
            conflicts=conflicts,
            raw_data_summary=str(tool_input.get("raw_data_summary", "")),
        )

    # ─── Run (disk cache only when enabled) ──────────────────────────────────

    def run(self) -> AgentSubReport:
        """Execute the full agent pipeline.

        Overrides BaseAgent.run(). Caching is off by default, so every run
        gathers live data. With DD_USE_CACHE=1 a previous report for the same
        company+ticker is reused and new reports are saved.
        """
        if not cache_enabled():
            return super().run()

        cache_path = self._report_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            self.logger.info(
                "Cache hit — loading %s report from %s", self.agent_name, cache_path.name
            )
            with open(cache_path) as f:
                return AgentSubReport.model_validate(json.load(f))

        report = super().run()

        with open(cache_path, "w") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2)
        self.logger.info("Cached %s report → %s", self.agent_name, cache_path.name)

        return report
