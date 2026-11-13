"""Tests for the Claude CLI wrapper, the MCP tool server, and the cache switch.

No test here starts the real `claude` binary: subprocess.run and run_claude are mocked.
"""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

os.environ["NEWS_API_KEY"] = "test-news-api-key"
os.environ["SEC_EDGAR_USER_AGENT"] = "TestBot test@example.com"

from src.models.schemas import AgentSubReport, InvestmentMemo
from src.utils import claude_cli
from src.utils.claude_cli import (
    ClaudeCLIError,
    ClaudeResult,
    build_command,
    extract_json_object,
    mcp_config_for_agent,
    parse_stream,
    run_claude,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _stream(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


def _assistant(*blocks: dict) -> dict:
    return {"type": "assistant", "message": {"content": list(blocks)}}


def _tool_result(tool_use_id: str, text: str, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": [{"type": "text", "text": text}]}
    if is_error:
        block["is_error"] = True
    return {"type": "user", "message": {"content": [block]}}


def _result(**overrides) -> dict:
    base = {
        "type": "result", "subtype": "success", "is_error": False, "result": "done",
        "num_turns": 3, "duration_ms": 1200, "total_cost_usd": 0.02,
    }
    base.update(overrides)
    return base


@pytest.fixture
def fake_cli(monkeypatch):
    """Pretend the claude binary exists at a fixed path."""
    monkeypatch.setattr(claude_cli, "find_claude_cli", lambda: "/fake/claude")


# ─── parse_stream ─────────────────────────────────────────────────────────────

class TestParseStream:
    def test_pairs_tool_calls_with_results(self):
        out = _stream(
            {"type": "system", "subtype": "init"},
            _assistant(
                {"type": "text", "text": "Looking up filings."},
                {"type": "tool_use", "id": "toolu_A1", "name": "mcp__deepdiligence__get_filings", "input": {"ticker": "AAPL"}},
            ),
            _tool_result("toolu_A1", json.dumps([{"form": "10-K"}])),
            _assistant({"type": "tool_use", "id": "toolu_B2", "name": "WebSearch", "input": {"query": "apple revenue"}}),
            _tool_result("toolu_B2", "Web search results: plain text"),
            _result(result="Summary of findings"),
        )
        parsed = parse_stream(out)

        assert parsed.is_error is False
        assert parsed.text == "Summary of findings"
        assert parsed.num_turns == 3
        assert [e["tool"] for e in parsed.tool_events] == ["get_filings", "WebSearch"]
        # MCP prefix stripped; JSON results parsed, plain text kept as text
        assert parsed.tool_events[0]["input"] == {"ticker": "AAPL"}
        assert parsed.tool_events[0]["output"] == [{"form": "10-K"}]
        assert parsed.tool_events[1]["output"] == "Web search results: plain text"

    def test_structured_output_is_returned_and_bookkeeping_tool_hidden(self):
        out = _stream(
            _assistant({"type": "tool_use", "id": "toolu_S", "name": "StructuredOutput", "input": {"a": 1}}),
            _tool_result("toolu_S", "ok"),
            _result(structured_output={"a": 1}, result='{"a":1}'),
        )
        parsed = parse_stream(out)
        assert parsed.structured == {"a": 1}
        assert parsed.tool_events == []

    def test_tool_error_is_wrapped(self):
        out = _stream(
            _assistant({"type": "tool_use", "id": "t1", "name": "WebFetch", "input": {"url": "https://x"}}),
            _tool_result("t1", "403 Forbidden", is_error=True),
            _result(),
        )
        assert parse_stream(out).tool_events[0]["output"] == {"error": "403 Forbidden"}

    def test_error_result_is_reported(self):
        parsed = parse_stream(_stream(_result(is_error=True, subtype="error_during_execution", result="Credit balance is too low")))
        assert parsed.is_error is True
        assert "Credit balance" in parsed.error

    def test_ignores_garbage_lines_and_falls_back_to_assistant_text(self):
        out = "not json\n" + _stream(_assistant({"type": "text", "text": "partial answer"}))
        parsed = parse_stream(out)
        assert parsed.text == "partial answer"
        assert parsed.is_error is False


# ─── extract_json_object ──────────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json_with_prose(self):
        text = 'Here you go:\n```json\n{"a": {"b": "has } brace"}}\n```\nThanks'
        assert extract_json_object(text) == {"a": {"b": "has } brace"}}

    def test_no_json(self):
        assert extract_json_object("nothing here") is None
        assert extract_json_object("") is None


# ─── build_command ────────────────────────────────────────────────────────────

class TestBuildCommand:
    def test_structured_call_has_no_tools(self, fake_cli):
        cmd = build_command(model="claude-opus-5-5", system_prompt="sys", json_schema={"type": "object"})
        assert cmd[0] == "/fake/claude"
        assert "--print" in cmd
        assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
        assert cmd[cmd.index("--system-prompt") + 1] == "sys"
        assert cmd[cmd.index("--tools") + 1] == ""
        assert json.loads(cmd[cmd.index("--json-schema") + 1]) == {"type": "object"}
        assert "--allowedTools" not in cmd
        assert "--mcp-config" not in cmd

    def test_research_call_allows_only_named_tools(self, fake_cli):
        cmd = build_command(
            model="m", system_prompt="sys",
            builtin_tools=("WebSearch", "WebFetch"),
            mcp_tools=["get_filings", "scrape_url"],
            mcp_config=mcp_config_for_agent("financial_analyst"),
        )
        assert cmd[cmd.index("--tools") + 1] == "WebSearch,WebFetch"
        assert cmd[cmd.index("--allowedTools") + 1] == (
            "WebSearch,WebFetch,mcp__deepdiligence__get_filings,mcp__deepdiligence__scrape_url"
        )
        config = json.loads(cmd[cmd.index("--mcp-config") + 1])
        server = config["mcpServers"]["deepdiligence"]
        assert server["args"] == ["-m", "src.tools.mcp_server", "--agent", "financial_analyst"]

    def test_isolation_and_safety_flags(self, fake_cli):
        cmd = build_command(model="m", system_prompt="sys")
        for flag in ("--restricted", "--disable-slash-commands", "--strict-mcp-config", "--no-session-persistence"):
            assert flag in cmd
        assert cmd[cmd.index("--permission-prompts") + 1] == "none"
        # Never bypass permissions, and never use --safe-mode (it drops the MCP data tools)
        assert "--dangerously-skip-permissions" not in cmd
        assert "--safe-mode" not in cmd


# ─── run_claude ───────────────────────────────────────────────────────────────

class TestRunClaude:
    def test_success_sends_prompt_on_stdin(self, fake_cli):
        completed = MagicMock(returncode=0, stdout=_stream(_result(structured_output={"ok": True})), stderr="")
        with patch("src.utils.claude_cli.subprocess.run", return_value=completed) as run:
            result = run_claude("the prompt", system_prompt="sys", model="m", json_schema={"type": "object"})

        assert result.structured == {"ok": True}
        kwargs = run.call_args.kwargs
        assert kwargs["input"] == "the prompt"
        assert "CLAUDECODE" not in kwargs["env"]
        assert "the prompt" not in run.call_args.args[0]  # prompt is never an argv element

    def test_nonzero_exit_raises(self, fake_cli):
        completed = MagicMock(returncode=1, stdout="", stderr="Not logged in")
        with patch("src.utils.claude_cli.subprocess.run", return_value=completed):
            with pytest.raises(ClaudeCLIError, match="Not logged in"):
                run_claude("p", system_prompt="s", model="m")

    def test_timeout_keeps_partial_tool_events(self, fake_cli):
        partial = _stream(
            _assistant({"type": "tool_use", "id": "t1", "name": "WebSearch", "input": {"query": "q"}}),
            _tool_result("t1", "results"),
        )
        exc = subprocess.TimeoutExpired(cmd="claude", timeout=5, output=partial.encode())
        with patch("src.utils.claude_cli.subprocess.run", side_effect=exc):
            result = run_claude("p", system_prompt="s", model="m", timeout=5, raise_on_error=False)

        assert result.is_error is True
        assert "timed out" in result.error
        assert len(result.tool_events) == 1

    def test_missing_binary_raises(self, monkeypatch):
        monkeypatch.setattr(claude_cli, "CLAUDE_CLI_PATH", "")
        monkeypatch.setattr(claude_cli.shutil, "which", lambda _: None)
        monkeypatch.setattr(claude_cli, "_FALLBACK_CLI_PATHS", ())
        with pytest.raises(ClaudeCLIError, match="Claude CLI not found"):
            claude_cli.find_claude_cli()


# ─── ReactAgent on top of the CLI ─────────────────────────────────────────────

class TestReactAgentWithCLI:
    def _agent(self):
        from src.agents.financial import FinancialAgent
        return FinancialAgent("Apple Inc", "AAPL")

    def test_research_collects_tool_results_and_uses_live_tools(self):
        events = [
            {"id": "toolu_0000AAAA1111", "tool": "get_revenue_data", "input": {"ticker": "AAPL"}, "output": [{"value": 1}]},
            {"id": "toolu_0000BBBB2222", "tool": "WebSearch", "input": {"query": "apple"}, "output": "results"},
        ]
        fake = ClaudeResult(text="summary", tool_events=events, num_turns=3)
        with patch("src.agents.react_agent.run_claude", return_value=fake) as run:
            collected = self._agent().gather_data()

        assert list(collected) == ["get_revenue_data_AAAA1111", "WebSearch_BBBB2222"]
        assert collected["get_revenue_data_AAAA1111"]["output"] == [{"value": 1}]

        kwargs = run.call_args.kwargs
        assert kwargs["builtin_tools"] == ("WebSearch", "WebFetch")
        assert kwargs["mcp_tools"] == ["get_revenue_data", "get_filings", "scrape_url"]
        assert "LIVE DATA RULES" in kwargs["system_prompt"]
        assert "Today's date:" in kwargs["prompt"]
        assert kwargs.get("json_schema") is None

    def test_research_failure_with_no_data_raises(self):
        failed = ClaudeResult(is_error=True, error="claude timed out after 900s")
        with patch("src.agents.react_agent.run_claude", return_value=failed):
            with pytest.raises(ClaudeCLIError, match="timed out"):
                self._agent().gather_data()

    def test_analysis_builds_report_from_structured_output(self):
        payload = {
            "sources": [{"id": "s1", "url": "https://sec.gov/x", "title": "10-K", "snippet": "Revenue 478.8B", "source_type": "sec_filing"}],
            "findings": [{"text": "FY2026 revenue was $478.8B", "source_ids": ["s1", "ghost"], "confidence": 0.9}],
            "conflicts": [],
            "confidence_score": 0.85,
            "raw_data_summary": "One filing reviewed.",
        }
        with patch("src.agents.react_agent.run_claude", return_value=ClaudeResult(structured=payload)) as run:
            report = self._agent().analyze({"k": {"tool": "get_filings", "input": {}, "output": []}})

        assert isinstance(report, AgentSubReport)
        assert report.findings[0].source_ids == ["s1"]  # unknown source id dropped
        assert report.confidence_score == 0.85
        kwargs = run.call_args.kwargs
        assert set(kwargs["json_schema"]["required"]) >= {"findings", "sources", "confidence_score"}
        assert not kwargs.get("builtin_tools")  # analysis step has no tools

    def test_analysis_failure_returns_zero_confidence_report(self):
        failed = ClaudeResult(is_error=True, error="boom")
        with patch("src.agents.react_agent.run_claude", return_value=failed):
            report = self._agent().analyze({})
        assert report.confidence_score == 0.0
        assert report.findings == []
        assert "boom" in report.raw_data_summary

    def test_run_ignores_existing_cache_file_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.utils.config.USE_CACHE", False)
        monkeypatch.setattr("src.agents.react_agent.CACHE_DIR", tmp_path)
        agent = self._agent()

        stale = AgentSubReport(agent_name="financial_analyst", confidence_score=0.11, raw_data_summary="STALE")
        path = agent._report_cache_path()
        path.parent.mkdir(parents=True)
        path.write_text(stale.model_dump_json())
        before = path.read_text()

        fresh = AgentSubReport(agent_name="financial_analyst", confidence_score=0.77, raw_data_summary="FRESH")
        with patch.object(type(agent), "gather_data", return_value={}), \
             patch.object(type(agent), "analyze", return_value=fresh):
            report = agent.run()

        assert report.raw_data_summary == "FRESH"
        assert path.read_text() == before  # cache neither read nor overwritten

    def test_run_uses_cache_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.utils.config.USE_CACHE", True)
        monkeypatch.setattr("src.agents.react_agent.CACHE_DIR", tmp_path)
        agent = self._agent()
        cached = AgentSubReport(agent_name="financial_analyst", confidence_score=0.5, raw_data_summary="CACHED")
        path = agent._report_cache_path()
        path.parent.mkdir(parents=True)
        path.write_text(cached.model_dump_json())

        with patch.object(type(agent), "gather_data", side_effect=AssertionError("should not run")):
            assert agent.run().raw_data_summary == "CACHED"


# ─── Financial tool: newest revenue first ─────────────────────────────────────

class TestRevenueOrdering:
    def test_revenue_tool_returns_newest_periods_first_and_caps(self):
        from src.agents.financial import _MAX_REVENUE_ENTRIES, FinancialAgent

        entries = [
            {"period": f"{y}-01-01 to {y}-12-31", "value": y, "unit": "USD", "filed_date": f"{y + 1}-02-01", "form": "10-K"}
            for y in range(1960, 2026)
        ]
        with patch("src.agents.financial.get_revenue_data", return_value=entries):
            out = FinancialAgent("X", "X")._execute_tool("get_revenue_data", {"ticker": "X"})

        assert len(out) == _MAX_REVENUE_ENTRIES
        assert out[0]["value"] == 2025
        assert out[-1]["value"] == 2025 - _MAX_REVENUE_ENTRIES + 1


# ─── MCP tool server ──────────────────────────────────────────────────────────

class TestMcpServer:
    def test_each_agent_exposes_its_own_tools(self):
        from src.tools.mcp_server import load_agent, tool_definitions

        expected = {
            "financial_analyst": ["get_revenue_data", "get_filings", "scrape_url"],
            "team_culture": ["scrape_careers_page", "scrape_url"],
            "market_competitive": ["fetch_company_news", "scrape_url"],
            "risk_sentiment": ["fetch_company_news", "scrape_url"],
        }
        for agent_name, names in expected.items():
            tools = tool_definitions(load_agent(agent_name))
            assert [t.name for t in tools] == names
            assert all(t.inputSchema["type"] == "object" for t in tools)

    def test_unknown_agent_rejected(self):
        from src.tools.mcp_server import load_agent
        with pytest.raises(ValueError, match="Unknown agent"):
            load_agent("quantitative_momentum")

    def test_tool_errors_are_returned_not_raised(self):
        from src.tools.mcp_server import execute_tool, load_agent

        agent = load_agent("financial_analyst")
        with patch("src.agents.financial.get_filings", side_effect=RuntimeError("EDGAR 503")):
            text = execute_tool(agent, "get_filings", {"ticker": "AAPL"})
        assert json.loads(text) == {"error": "RuntimeError: EDGAR 503"}

        assert "unknown tool" in json.loads(execute_tool(agent, "nope", {}))["error"]

    def test_tool_output_is_capped(self):
        from src.tools.mcp_server import MAX_RESULT_CHARS, execute_tool, load_agent

        agent = load_agent("market_competitive")
        huge = {"url": "https://x", "title": "t", "text": "y" * 50000, "error": None}
        with patch("src.agents.market.scrape_url", return_value=huge):
            text = execute_tool(agent, "scrape_url", {"url": "https://x"})
        assert len(text) <= MAX_RESULT_CHARS


# ─── Synthesis, recommender and batch runner on top of the CLI ────────────────

class TestSynthesisAndBatch:
    def _sub_reports(self) -> list[AgentSubReport]:
        return [AgentSubReport(agent_name="financial_analyst", confidence_score=0.8, raw_data_summary="ok")]

    def test_synthesis_uses_structured_output(self):
        from src.agents.synthesis import SynthesisAgent

        payload = {
            "executive_summary": "Summary.",
            "sections": [{"title": "Financial Analysis", "content": "Body", "confidence_score": 0.7, "key_claims": []}],
            "overall_confidence": 0.72,
        }
        with patch("src.agents.synthesis.run_claude", return_value=ClaudeResult(structured=payload)) as run:
            memo = SynthesisAgent("TestCo").synthesize(self._sub_reports())

        assert memo.executive_summary == "Summary."
        assert memo.overall_confidence == 0.72
        assert "executive_summary" in run.call_args.kwargs["json_schema"]["properties"]

    def test_synthesis_falls_back_to_json_in_text(self):
        from src.agents.synthesis import SynthesisAgent

        text = '```json\n{"executive_summary": "From text.", "sections": [], "overall_confidence": 0.4}\n```'
        with patch("src.agents.synthesis.run_claude", return_value=ClaudeResult(text=text)):
            memo = SynthesisAgent("TestCo").synthesize(self._sub_reports())
        assert memo.executive_summary == "From text."

    def test_synthesis_failure_returns_empty_memo(self):
        from src.agents.synthesis import SynthesisAgent

        with patch("src.agents.synthesis.run_claude", return_value=ClaudeResult(is_error=True, error="boom")):
            memo = SynthesisAgent("TestCo").synthesize(self._sub_reports())
        assert memo.overall_confidence == 0.0
        assert memo.metadata["error"] == "boom"

    def test_recommender_builds_report_from_structured_output(self):
        from src.scheduler.recommender import RecommendationEngine

        memo = InvestmentMemo(company_name="Apple Inc", executive_summary="Strong.", overall_confidence=0.8)
        payload = {
            "ratings": [{
                "company_name": "Apple Inc", "ticker": "AAPL", "sector": "Technology", "rating": "BUY",
                "rank": 1, "bull_case": "b", "bear_case": "r", "rationale": "x",
                "suggested_weight_pct": 100, "confidence": 0.8,
            }],
            "top_picks": ["AAPL"], "avoid": [], "macro_commentary": "m", "sector_views": {"Technology": "Bullish"},
        }
        with patch("src.scheduler.recommender.run_claude", return_value=ClaudeResult(structured=payload)):
            report = RecommendationEngine().generate({"AAPL": memo})

        assert report.ratings[0].ticker == "AAPL"
        assert report.top_picks == ["AAPL"]
        assert report.model_used.startswith("claude")

    def test_batch_runner_ignores_cached_memo_by_default(self, monkeypatch):
        from src.scheduler import batch_runner

        monkeypatch.setattr("src.utils.config.USE_CACHE", False)
        stale = InvestmentMemo(company_name="Apple Inc", executive_summary="STALE", overall_confidence=0.1)
        fresh = InvestmentMemo(company_name="Apple Inc", executive_summary="FRESH", overall_confidence=0.9)

        orchestrator = MagicMock()
        orchestrator.return_value.run.return_value = fresh
        with patch.object(batch_runner, "load_cached_memo", return_value=stale) as load, \
             patch.object(batch_runner, "save_memo") as save, \
             patch.object(batch_runner, "Orchestrator", orchestrator):
            _, memo = batch_runner._run_one({"company": "Apple Inc", "ticker": "AAPL"}, force_refresh=False)

        assert memo.executive_summary == "FRESH"
        load.assert_not_called()
        save.assert_not_called()
