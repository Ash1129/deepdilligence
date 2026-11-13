"""Thin wrapper around the Claude Code CLI in headless mode (`claude -p`).

Every LLM call in DeepDiligence goes through `run_claude()`. Two shapes are used:

* **Research call** — the CLI runs its own agent loop with a restricted tool set
  (the calling agent's data tools, served over MCP, plus built-in web search).
  Every tool call and result is captured so the analysis step can cite sources.
* **Structured call** — no tools; the reply is constrained to a JSON Schema via
  `--json-schema` and returned as a dict.

The CLI is started with `--restricted`, `--disable-slash-commands` and
`--strict-mcp-config` so the user's own settings, plugins, skills and MCP servers
never leak into a pipeline run, with an explicit `--tools` list so the model can
only search the web and call this project's read-only data tools, and with
`--permission-prompts none` so nothing can block waiting for approval.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config import CLAUDE_CLI_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)

MCP_SERVER_NAME = "deepdiligence"
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

# Built-in CLI tools research agents may use to reach live data.
WEB_TOOLS: tuple[str, ...] = ("WebSearch", "WebFetch")

_FALLBACK_CLI_PATHS = (
    "~/.local/bin/claude",
    "~/.claude/local/claude",
    "/opt/homebrew/bin/claude",
    "/usr/local/bin/claude",
)


class ClaudeCLIError(RuntimeError):
    """Raised when the Claude CLI is missing, times out, or reports an error."""


@dataclass
class ClaudeResult:
    """Parsed outcome of one `claude -p` invocation."""

    text: str = ""
    structured: dict[str, Any] | None = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    is_error: bool = False
    error: str = ""
    num_turns: int = 0
    duration_ms: int = 0
    cost_usd: float = 0.0


def find_claude_cli() -> str:
    """Locate the `claude` binary, or raise ClaudeCLIError with a fix-it hint."""
    if CLAUDE_CLI_PATH:
        if Path(CLAUDE_CLI_PATH).expanduser().exists():
            return str(Path(CLAUDE_CLI_PATH).expanduser())
        raise ClaudeCLIError(f"CLAUDE_CLI_PATH points to a missing file: {CLAUDE_CLI_PATH}")

    found = shutil.which("claude")
    if found:
        return found

    for candidate in _FALLBACK_CLI_PATHS:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)

    raise ClaudeCLIError(
        "Claude CLI not found. Install Claude Code and run `claude` once to log in, "
        "or set CLAUDE_CLI_PATH to the binary."
    )


def mcp_config_for_agent(agent_name: str) -> dict[str, Any]:
    """MCP config that serves one agent's data tools from this project."""
    return {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "command": sys.executable,
                "args": ["-m", "src.tools.mcp_server", "--agent", agent_name],
                "env": {"PYTHONPATH": str(PROJECT_ROOT)},
            }
        }
    }


def build_command(
    *,
    model: str,
    system_prompt: str,
    json_schema: dict[str, Any] | None = None,
    builtin_tools: tuple[str, ...] | list[str] = (),
    mcp_tools: tuple[str, ...] | list[str] = (),
    mcp_config: dict[str, Any] | None = None,
) -> list[str]:
    """Assemble the argv for one headless CLI call (the prompt goes on stdin)."""
    cmd: list[str] = [
        find_claude_cli(),
        "--print",
        "--model", model,
        "--output-format", "stream-json",
        "--verbose",
        "--system-prompt", system_prompt,
        # Isolation: ignore the user's settings, plugins, skills and MCP servers so
        # a pipeline run sees only what this project hands it. (--safe-mode is not
        # used because it also drops the --mcp-config data tools.)
        "--restricted",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--permission-prompts", "none",
        # Built-in tools available to the model; "" disables all of them.
        "--tools", ",".join(builtin_tools),
    ]

    allowed = list(builtin_tools) + [f"{MCP_TOOL_PREFIX}{name}" for name in mcp_tools]
    if allowed:
        cmd += ["--allowedTools", ",".join(allowed)]
    if mcp_config:
        cmd += ["--mcp-config", json.dumps(mcp_config)]
    if json_schema:
        cmd += ["--json-schema", json.dumps(json_schema)]
    return cmd


def run_claude(
    prompt: str,
    *,
    system_prompt: str,
    model: str,
    json_schema: dict[str, Any] | None = None,
    builtin_tools: tuple[str, ...] | list[str] = (),
    mcp_tools: tuple[str, ...] | list[str] = (),
    mcp_config: dict[str, Any] | None = None,
    timeout: int = 600,
    raise_on_error: bool = True,
) -> ClaudeResult:
    """Run one headless Claude CLI call and return its parsed result.

    Args:
        prompt: User message, sent on stdin.
        system_prompt: Replaces the CLI's default system prompt.
        model: Model ID or alias passed to `--model`.
        json_schema: If given, the reply is constrained to this schema and
            returned in `ClaudeResult.structured`.
        builtin_tools: Built-in CLI tools to enable (e.g. WebSearch, WebFetch).
        mcp_tools: Bare names of MCP tools to allow from `mcp_config`.
        mcp_config: MCP server definition (see `mcp_config_for_agent`).
        timeout: Wall-clock limit in seconds.
        raise_on_error: If False, failures are returned in the result (with any
            tool events captured before the failure) instead of raised.

    Raises:
        ClaudeCLIError: On a missing binary, timeout, or CLI-reported error,
            unless `raise_on_error` is False.
    """
    cmd = build_command(
        model=model,
        system_prompt=system_prompt,
        json_schema=json_schema,
        builtin_tools=builtin_tools,
        mcp_tools=mcp_tools,
        mcp_config=mcp_config,
    )

    # CLAUDECODE marks "already inside a Claude Code session"; drop it so the
    # backend also works when it was started from one.
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    stdout = ""
    stderr = ""
    failure = ""
    # A neutral working directory keeps project files out of the model's reach.
    with tempfile.TemporaryDirectory(prefix="deepdiligence-claude-") as workdir:
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                env=env,
            )
            stdout, stderr = proc.stdout or "", proc.stderr or ""
            if proc.returncode != 0:
                failure = f"claude exited with code {proc.returncode}: {stderr.strip()[:500]}"
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            failure = f"claude timed out after {timeout}s"
        except OSError as exc:
            failure = f"could not start claude: {exc}"

    result = parse_stream(stdout)
    if failure and not result.error:
        result.is_error = True
        result.error = failure
    elif not failure and not result.is_error and not stdout.strip():
        result.is_error = True
        result.error = "claude produced no output"

    logger.info(
        "claude call done — model=%s turns=%d tools=%d %.1fs%s",
        model,
        result.num_turns,
        len(result.tool_events),
        result.duration_ms / 1000,
        f" ERROR: {result.error}" if result.is_error else "",
    )

    if result.is_error and raise_on_error:
        raise ClaudeCLIError(result.error or "claude call failed")
    return result


# ─── Stream parsing ───────────────────────────────────────────────────────────

def parse_stream(stdout: str) -> ClaudeResult:
    """Parse `--output-format stream-json` output into a ClaudeResult.

    Pairs every tool_use block with its tool_result so callers get an ordered
    list of {id, tool, input, output} events, and reads the final `result` line
    for the reply text, structured output and run statistics.
    """
    result = ClaudeResult()
    pending: dict[str, dict[str, Any]] = {}
    text_parts: list[str] = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind = event.get("type")
        if kind == "assistant":
            for block in _content_blocks(event):
                if block.get("type") == "tool_use":
                    tool_event = {
                        "id": str(block.get("id", "")),
                        "tool": _strip_mcp_prefix(str(block.get("name", ""))),
                        "input": block.get("input") or {},
                        "output": None,
                    }
                    pending[tool_event["id"]] = tool_event
                    result.tool_events.append(tool_event)
                elif block.get("type") == "text" and block.get("text"):
                    text_parts.append(str(block["text"]))
        elif kind == "user":
            for block in _content_blocks(event):
                if block.get("type") == "tool_result":
                    tool_event = pending.get(str(block.get("tool_use_id", "")))
                    if tool_event is not None:
                        tool_event["output"] = _tool_result_payload(block)
        elif kind == "result":
            result.is_error = bool(event.get("is_error"))
            result.num_turns = int(event.get("num_turns") or 0)
            result.duration_ms = int(event.get("duration_ms") or 0)
            result.cost_usd = float(event.get("total_cost_usd") or 0.0)
            final_text = event.get("result")
            if isinstance(final_text, str):
                result.text = final_text
            structured = event.get("structured_output")
            if isinstance(structured, dict):
                result.structured = structured
            if result.is_error:
                result.error = str(final_text or event.get("subtype") or "claude reported an error")[:500]

    if not result.text and text_parts:
        result.text = "\n".join(text_parts)

    # Internal bookkeeping tools (e.g. the structured-output tool) are not data sources.
    result.tool_events = [e for e in result.tool_events if e["tool"] != "StructuredOutput"]
    return result


def _content_blocks(event: dict[str, Any]) -> list[dict[str, Any]]:
    content = (event.get("message") or {}).get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _strip_mcp_prefix(name: str) -> str:
    return name[len(MCP_TOOL_PREFIX):] if name.startswith(MCP_TOOL_PREFIX) else name


def _tool_result_payload(block: dict[str, Any]) -> Any:
    """Return a tool result as parsed JSON when possible, else as text."""
    content = block.get("content")
    if isinstance(content, list):
        text = "\n".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = "" if content is None else str(content)

    try:
        payload: Any = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        payload = text

    if block.get("is_error"):
        return {"error": payload}
    return payload


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from free text.

    Used only as a fallback when the CLI did not return `structured_output`.
    """
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None
