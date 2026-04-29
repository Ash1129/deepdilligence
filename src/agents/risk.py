"""Risk & Sentiment agent — litigation, regulatory, reputational, and financial risks."""

import logging
from typing import Any

from src.agents.react_agent import ReactAgent
from src.data.news import fetch_company_news
from src.data.web import scrape_url
from src.models.prompts import RISK_ANALYZE_SYSTEM, RISK_REACT_SYSTEM
from src.models.schemas import AgentSubReport

logger = logging.getLogger(__name__)

# Risk-focused news query templates to catch different risk categories
_RISK_QUERIES = [
    "{company} lawsuit",
    "{company} investigation",
    "{company} layoffs",
    "{company} controversy",
    "{company} breach data",
    "{company} SEC enforcement",
    "{company} fraud",
    "{company} recall",
]

# Tool definitions in OpenAI format
_RISK_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_company_news",
            "description": (
                "Fetch recent news articles about a company from NewsAPI. "
                "For risk analysis, use targeted queries like '[company] lawsuit', "
                "'[company] investigation', '[company] layoffs', '[company] breach'. "
                "Results are cached — multiple queries are okay but be selective."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": (
                            "Search query — can include risk keywords, e.g. "
                            "'Apple lawsuit' or 'Tesla SEC investigation'"
                        ),
                    },
                    "days_back": {
                        "type": "integer",
                        "description": "How many days back to search (default 60 for risks — longer lookback)",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Max articles to return (default 20)",
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
                "Use for SEC EDGAR enforcement releases, court filings, "
                "regulatory agency pages, or specific news articles about risks."
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


class RiskAgent(ReactAgent):
    """Analyst for risks: legal, regulatory, reputational, financial, and leadership.

    Data sources:
    - NewsAPI with risk-targeted queries (lawsuits, investigations, layoffs, breaches)
    - SEC EDGAR enforcement releases (via web scraping)
    - Regulatory agency pages
    - Specific news articles identified during research

    This agent deliberately searches for negative signals. It reports what it finds —
    including the absence of significant risk signals when the search comes up clean.
    """

    @property
    def agent_name(self) -> str:
        return "risk_sentiment"

    @property
    def description(self) -> str:
        return (
            "Surfaces legal, regulatory, reputational, financial, and leadership risks "
            "through targeted news analysis and regulatory source review."
        )

    def _execute_tool(self, name: str, tool_input: dict) -> Any:
        """Route a Claude tool call to the appropriate data function.

        Args:
            name: Tool name as defined in _RISK_TOOLS.
            tool_input: Parameters dict from the Claude tool call.

        Returns:
            JSON-serializable result from the underlying data function.
        """
        if name == "fetch_company_news":
            articles = fetch_company_news(
                company_name=tool_input["company_name"],
                days_back=int(tool_input.get("days_back", 60)),
                page_size=int(tool_input.get("page_size", 20)),
            )
            # Trim content but keep more than other agents — details matter for risk
            trimmed = []
            for article in articles:
                a = dict(article)
                if a.get("content"):
                    a["content"] = a["content"][:800]
                trimmed.append(a)
            return trimmed

        if name == "scrape_url":
            result = scrape_url(tool_input["url"])
            if result.get("text"):
                result = dict(result)
                result["text"] = result["text"][:5000]
            return result

        raise ValueError(f"RiskAgent: unknown tool '{name}'")

    def gather_data(self) -> dict[str, Any]:
        """Run the ReAct loop to surface risk signals.

        Instructs the LLM to run multiple targeted news queries (lawsuits,
        investigations, layoffs, breaches) and scrape any specific sources
        that look like material risk disclosures.
        """
        # Build example risk queries for the initial prompt
        risk_queries_formatted = "\n".join(
            f"  - '{q.format(company=self.company_name)}'"
            for q in _RISK_QUERIES[:4]  # Show a few as examples
        )

        initial_message = (
            f"Conduct risk and sentiment due diligence on: {self.company_name}\n\n"
            "Your goal is to surface ALL material risks an investor should know about.\n\n"
            "Use targeted news queries to cover each risk category:\n"
            f"{risk_queries_formatted}\n"
            "  - (and other risk-specific queries as needed)\n\n"
            "After news queries, scrape any specific articles or regulatory pages "
            "that look like they contain material risk disclosures. "
            "Also check https://efts.sec.gov/LATEST/search-index?q=%22"
            f"{self.company_name.replace(' ', '%20')}%22&dateRange=custom&startdt=2023-01-01"
            " for any SEC enforcement actions.\n\n"
            "Cover at minimum: legal risks, regulatory risks, reputational risks, "
            "financial risk signals, and leadership/governance risks. "
            "If a category comes up clean, note that explicitly."
        )

        _, collected = self._run_react_loop(
            system_prompt=RISK_REACT_SYSTEM,
            initial_user_message=initial_message,
            tools=_RISK_TOOLS,
            tool_executor=self._execute_tool,
        )
        return collected

    def analyze(self, raw_data: dict[str, Any]) -> AgentSubReport:
        """Produce the structured risk sub-report from gathered data.

        Args:
            raw_data: Output of gather_data() — dict of tool results.

        Returns:
            AgentSubReport with risk findings, sourced claims, and confidence scores.
        """
        return self._produce_structured_report(RISK_ANALYZE_SYSTEM, raw_data)
