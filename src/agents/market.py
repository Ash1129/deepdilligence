"""Market & Competitive agent — competitors, positioning, and growth signals."""

import logging
from typing import Any

from src.agents.react_agent import ReactAgent
from src.data.news import fetch_company_news
from src.data.web import scrape_url
from src.models.prompts import MARKET_ANALYZE_SYSTEM, MARKET_REACT_SYSTEM
from src.models.schemas import AgentSubReport

logger = logging.getLogger(__name__)

# Tool definitions in OpenAI format
_MARKET_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_company_news",
            "description": (
                "Fetch recent news articles about a company from NewsAPI. "
                "Returns: title, description, url, source, published_at, content for each article. "
                "Use for competitive intelligence, market positioning, partnership announcements, "
                "and analyst coverage. Be mindful of the 100 req/day limit — results are cached."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Company name to search for (e.g. 'Apple Inc')",
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "How many days back to search (default 30, max 90)",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Max articles to return (default 20, max 100)",
                    },
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": (
                "Fetch and extract text from any URL. "
                "Use for company product/pricing pages, competitor websites, "
                "market research reports, or industry analyst articles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch (https://...)",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


class MarketAgent(ReactAgent):
    """Analyst for market positioning, competitive landscape, and growth signals.

    Data sources:
    - NewsAPI (competitive intelligence, analyst coverage, market news)
    - Company website (product pages, pricing, customer logos, use cases)
    - Competitor websites (for positioning comparison)
    - Industry reports and analyst articles (via web scraping)
    """

    @property
    def agent_name(self) -> str:
        return "market_competitive"

    @property
    def description(self) -> str:
        return (
            "Maps the competitive landscape, assesses market positioning and differentiation, "
            "and identifies growth signals and competitive threats."
        )

    def _execute_tool(self, name: str, tool_input: dict) -> Any:
        """Route a Claude tool call to the appropriate data function.

        Args:
            name: Tool name as defined in _MARKET_TOOLS.
            tool_input: Parameters dict from the Claude tool call.

        Returns:
            JSON-serializable result from the underlying data function.
        """
        if name == "fetch_company_news":
            articles = fetch_company_news(
                company_name=tool_input["company_name"],
                days_back=int(tool_input.get("days_back", 30)),
                page_size=int(tool_input.get("page_size", 20)),
            )
            # Truncate content fields to keep context manageable
            trimmed = []
            for article in articles:
                a = dict(article)
                if a.get("content"):
                    a["content"] = a["content"][:500]
                trimmed.append(a)
            return trimmed

        if name == "scrape_url":
            result = scrape_url(tool_input["url"])
            if result.get("text"):
                result = dict(result)
                result["text"] = result["text"][:5000]
            return result

        raise ValueError(f"MarketAgent: unknown tool '{name}'")

    def gather_data(self) -> dict[str, Any]:
        """Run the ReAct loop to gather competitive and market intelligence.

        Starts with news (broad coverage), then scrapes the company website for
        product/pricing details and any competitor pages for positioning comparison.
        """
        initial_message = (
            f"Conduct market and competitive due diligence on: {self.company_name}\n\n"
            "Use the tools to gather:\n"
            "- Recent news about the company (look for competitor mentions, partnerships, market wins)\n"
            "- Company website product and pricing pages\n"
            "- Any competitor comparison articles or industry analyst coverage\n\n"
            f"Start by fetching news for '{self.company_name}'. "
            "Then scrape the company's main website and product pages. "
            "If you identify key competitors from the news, scrape one or two competitor sites "
            "for positioning comparison. "
            "Stop when you have identified the top competitors and the company's market position."
        )

        _, collected = self._run_react_loop(
            system_prompt=MARKET_REACT_SYSTEM,
            initial_user_message=initial_message,
            tools=_MARKET_TOOLS,
            tool_executor=self._execute_tool,
        )
        return collected

    def analyze(self, raw_data: dict[str, Any]) -> AgentSubReport:
        """Produce the structured market & competitive sub-report from gathered data.

        Args:
            raw_data: Output of gather_data() — dict of tool results.

        Returns:
            AgentSubReport with market findings, sourced claims, and confidence scores.
        """
        return self._produce_structured_report(MARKET_ANALYZE_SYSTEM, raw_data)
