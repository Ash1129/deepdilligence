"""MCP stdio server that exposes one specialist agent's data tools to the Claude CLI.

Started by the CLI itself (see `mcp_config_for_agent` in src/utils/claude_cli.py):

    python -m src.tools.mcp_server --agent financial_analyst

The tool definitions and the routing logic are not duplicated here. They are read
from the agent class (`TOOLS` and `_execute_tool`), so each agent file stays the
single source of truth for what its tools are and how they run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

# stdout carries the MCP protocol, so all logging must go to stderr.
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("deepdiligence.mcp")

# Cap on one tool result, in characters, to keep the research context focused.
MAX_RESULT_CHARS = 12000


def load_agent(agent_name: str) -> Any:
    """Instantiate the specialist agent whose tools this server should expose."""
    from src.agents.financial import FinancialAgent
    from src.agents.market import MarketAgent
    from src.agents.risk import RiskAgent
    from src.agents.team import TeamAgent

    classes = {
        "financial_analyst": FinancialAgent,
        "team_culture": TeamAgent,
        "market_competitive": MarketAgent,
        "risk_sentiment": RiskAgent,
    }
    if agent_name not in classes:
        raise ValueError(f"Unknown agent '{agent_name}'. Expected one of: {sorted(classes)}")
    return classes[agent_name]("mcp-tool-host")


def tool_definitions(agent: Any) -> list[types.Tool]:
    """Convert the agent's function-style tool definitions into MCP tools."""
    tools: list[types.Tool] = []
    for definition in agent.TOOLS:
        fn = definition["function"]
        tools.append(
            types.Tool(
                name=fn["name"],
                description=fn.get("description", ""),
                inputSchema=fn.get("parameters", {"type": "object", "properties": {}}),
            )
        )
    return tools


def execute_tool(agent: Any, name: str, arguments: dict[str, Any]) -> str:
    """Run one tool through the agent's router and return a JSON string."""
    try:
        result = agent._execute_tool(name, arguments or {})
    except Exception as exc:  # the model sees the error and can adapt
        logger.warning("Tool %s raised %s: %s", name, type(exc).__name__, exc)
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, default=str)[:MAX_RESULT_CHARS]


def build_server(agent: Any) -> Server:
    """Create the MCP server for one agent."""
    server: Server = Server("deepdiligence")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return tool_definitions(agent)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        # Data functions are blocking (requests); keep the event loop free.
        text = await asyncio.to_thread(execute_tool, agent, name, arguments)
        return [types.TextContent(type="text", text=text)]

    return server


async def serve(agent_name: str) -> None:
    """Run the stdio server until the CLI closes the pipe."""
    server = build_server(load_agent(agent_name))
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepDiligence data-tool MCP server")
    parser.add_argument("--agent", required=True, help="agent_name whose tools to expose")
    args = parser.parse_args()
    asyncio.run(serve(args.agent))


if __name__ == "__main__":
    main()
