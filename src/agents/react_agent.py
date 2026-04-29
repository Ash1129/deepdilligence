"""ReactAgent: intermediate base class providing the ReAct loop and structured output."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from openai import BadRequestError, OpenAI

from src.agents.base_agent import BaseAgent
from src.models.prompts import PRODUCE_ANALYSIS_TOOL
from src.models.schemas import (
    AgentClaim,
    AgentSubReport,
    ConflictingClaim,
    Source,
    SourceType,
)
from src.utils.config import AGENT_MODEL, CACHE_DIR, get_openai_api_key

logger = logging.getLogger(__name__)

MAX_REACT_ITERATIONS = 8


class ReactAgent(BaseAgent):
    """Extends BaseAgent with the ReAct pattern and disk-cached runs.

    Provides two shared capabilities for specialist agents:
    1. _run_react_loop(): LLM-driven adaptive data gathering via tool use.
    2. _produce_structured_report(): forces structured AgentSubReport via tool call.

    Specialist agents implement gather_data() and analyze() using these helpers,
    and define their own tools and tool executor.
    """

    def __init__(self, company_name: str, ticker: str | None = None, **kwargs: Any) -> None:
        super().__init__(company_name, ticker, **kwargs)
        self._client = OpenAI(api_key=get_openai_api_key())

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
        tools: list[dict],
        tool_executor: Callable[[str, dict], Any],
        max_iterations: int = MAX_REACT_ITERATIONS,
    ) -> tuple[list[dict], dict[str, Any]]:
        """Execute the ReAct pattern: Reason → Act (tool call) → Observe → repeat.

        Args:
            system_prompt: Agent's research system prompt.
            initial_user_message: Starting research request.
            tools: OpenAI-format tool definitions for this agent.
            tool_executor: Callable(tool_name, tool_input_dict) → result.
            max_iterations: Hard cap on LLM calls.

        Returns:
            Tuple of (conversation_messages, collected_data_dict).
        """
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_message},
        ]
        collected_data: dict[str, Any] = {}

        for iteration in range(max_iterations):
            self.logger.debug("ReAct iteration %d / %d", iteration + 1, max_iterations)

            try:
                response = self._client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_completion_tokens=4096,
                )
            except BadRequestError as e:
                # Safety filter (cyber_policy) or other non-retryable rejection —
                # end the loop gracefully rather than crashing the whole pipeline.
                error_code = getattr(e, "code", "unknown")
                self.logger.warning(
                    "LLM call rejected in ReAct loop (code=%s): %s — ending loop early",
                    error_code, str(e)[:200],
                )
                break

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Append assistant turn to history
            messages.append(message)

            if finish_reason == "stop":
                self.logger.debug("ReAct loop finished (stop) after %d iterations", iteration + 1)
                break

            if finish_reason == "tool_calls" and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_input = {}

                    self.logger.debug("Tool call: %s(%s)", tool_name, list(tool_input.keys()))

                    try:
                        result = tool_executor(tool_name, tool_input)
                    except Exception as exc:
                        self.logger.warning("Tool %s raised %s: %s", tool_name, type(exc).__name__, exc)
                        result = {"error": str(exc)}

                    # Store with a unique key so analyze() can enumerate sources
                    data_key = f"{tool_name}_{tool_call.id[:8]}"
                    collected_data[data_key] = {
                        "tool": tool_name,
                        "input": tool_input,
                        "output": result,
                    }

                    # Feed result back — OpenAI uses role="tool"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str)[:8000],
                    })
            else:
                self.logger.warning("Unexpected finish_reason: %s — ending loop", finish_reason)
                break

        self.logger.info(
            "%s: ReAct loop done — %d sources collected",
            self.agent_name,
            len(collected_data),
        )
        return messages, collected_data

    # ─── Structured output ───────────────────────────────────────────────────

    def _produce_structured_report(
        self, system_prompt: str, raw_data: dict[str, Any]
    ) -> AgentSubReport:
        """Call the LLM with gathered data and force a structured AgentSubReport.

        Uses tool_choice to guarantee the model returns JSON matching the schema.

        Args:
            system_prompt: Analysis system prompt for this agent.
            raw_data: Output of gather_data().

        Returns:
            Validated AgentSubReport instance.
        """
        data_text = self._format_raw_data(raw_data)

        user_message = (
            f"Company under analysis: {self.company_name}\n"
            f"Ticker: {self.ticker or 'not available (private or unknown)'}\n\n"
            "=== GATHERED DATA ===\n"
            f"{data_text}\n\n"
            "Analyze the data above. Call produce_analysis with your structured findings. "
            "Every finding must reference at least one source ID. Be specific — no vague claims."
        )

        response = self._client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=[PRODUCE_ANALYSIS_TOOL],
            tool_choice={"type": "function", "function": {"name": "produce_analysis"}},
            max_completion_tokens=8192,
        )

        message = response.choices[0].message
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "produce_analysis":
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        self.logger.error("Failed to parse produce_analysis arguments: %s", e)
                        break
                    return self._build_subreport(tool_input)

        # Should never reach here with forced tool_choice
        self.logger.error("produce_analysis not returned — building empty fallback report")
        return AgentSubReport(
            agent_name=self.agent_name,
            findings=[],
            sources=[],
            confidence_score=0.0,
            conflicts=[],
            raw_data_summary="Analysis step failed: produce_analysis tool was not called.",
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
                output_str = json.dumps(value["output"], default=str)
                if len(output_str) > 4000:
                    output_str = output_str[:4000] + "... [truncated]"
                parts.append(f"Output: {output_str}")
            else:
                raw_str = json.dumps(value, default=str)
                parts.append(raw_str[:4000] if len(raw_str) > 4000 else raw_str)
            parts.append("")

        return "\n".join(parts) if parts else "No data was gathered."

    def _build_subreport(self, tool_input: dict) -> AgentSubReport:
        """Convert the produce_analysis tool call input into a validated AgentSubReport."""
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

    # ─── Run with disk cache ─────────────────────────────────────────────────

    def run(self) -> AgentSubReport:
        """Execute the full agent pipeline with disk caching.

        Overrides BaseAgent.run(). On re-runs with the same company+ticker,
        returns the cached report instantly — no API calls made.
        """
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
