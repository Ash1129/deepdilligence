"""Financial Analyst agent — SEC EDGAR data, revenue trends, profitability signals."""

import logging
from typing import Any

from src.agents.react_agent import ReactAgent
from src.data.edgar import get_filings, get_revenue_data
from src.data.web import scrape_url
from src.models.prompts import FINANCIAL_ANALYZE_SYSTEM, FINANCIAL_REACT_SYSTEM
from src.models.schemas import AgentSubReport

logger = logging.getLogger(__name__)

# Tool definitions in OpenAI format
_FINANCIAL_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_revenue_data",
            "description": (
                "Fetch annual and quarterly revenue figures from SEC EDGAR XBRL data for a public company. "
                "Returns a list of {period, value, unit, filed_date, form} dicts. "
                "Use this first when a ticker is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL'",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_filings",
            "description": (
                "Fetch recent SEC filing metadata (10-K, 10-Q) for a public company. "
                "Returns filing dates, accession numbers, and direct URLs to primary documents. "
                "Use after get_revenue_data to get context from the actual filings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "filing_type": {
                        "type": "string",
                        "description": "Filing type: '10-K' (annual) or '10-Q' (quarterly)",
                        "enum": ["10-K", "10-Q"],
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of filings to return (1-5)",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": (
                "Fetch and extract text from any URL. "
                "Use for investor relations pages, annual report landing pages, "
                "press releases about financial results, or company home pages."
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


class FinancialAgent(ReactAgent):
    """Analyst for financial health: revenue, growth, profitability, and balance sheet.

    Data sources:
    - SEC EDGAR XBRL (structured revenue and filing data for public companies)
    - Company IR / investor relations pages
    - Press releases and financial news (via web scraping)

    For private companies the EDGAR tools will return empty results; the agent
    falls back to web scraping of press releases and news mentions.
    """

    @property
    def agent_name(self) -> str:
        return "financial_analyst"

    @property
    def description(self) -> str:
        return (
            "Analyzes financial health, revenue trends, profitability signals, "
            "and balance sheet quality from SEC filings and public disclosures."
        )

    def _execute_tool(self, name: str, tool_input: dict) -> Any:
        """Route a Claude tool call to the appropriate data function.

        Args:
            name: Tool name as defined in _FINANCIAL_TOOLS.
            tool_input: Parameters dict from the Claude tool call.

        Returns:
            JSON-serializable result from the underlying data function.
        """
        if name == "get_revenue_data":
            return get_revenue_data(tool_input["ticker"])

        if name == "get_filings":
            return get_filings(
                ticker=tool_input["ticker"],
                filing_type=tool_input.get("filing_type", "10-K"),
                count=int(tool_input.get("count", 3)),
            )

        if name == "scrape_url":
            result = scrape_url(tool_input["url"])
            # Trim scraped text to avoid bloating the context
            if result.get("text"):
                result = dict(result)
                result["text"] = result["text"][:5000]
            return result

        raise ValueError(f"FinancialAgent: unknown tool '{name}'")

    def gather_data(self) -> dict[str, Any]:
        """Run the ReAct loop to adaptively gather financial data.

        The LLM decides which tools to call and in what order based on what's
        available (ticker vs. no ticker) and what it finds along the way.
        """
        ticker_hint = (
            f"Stock ticker: {self.ticker}" if self.ticker
            else "No ticker available — this may be a private company."
        )

        initial_message = (
            f"Conduct financial due diligence on: {self.company_name}\n"
            f"{ticker_hint}\n\n"
            "Use the tools to gather:\n"
            "- Revenue figures (EDGAR XBRL if ticker available)\n"
            "- Recent 10-K / 10-Q filing metadata and URLs\n"
            "- Investor relations page content\n"
            "- Any press releases about financial performance\n\n"
            "Start with EDGAR structured data if the ticker is available, "
            "then scrape the company's IR page and any relevant press releases. "
            "Stop when you have 3-4 solid financial data points."
        )

        _, collected = self._run_react_loop(
            system_prompt=FINANCIAL_REACT_SYSTEM,
            initial_user_message=initial_message,
            tools=_FINANCIAL_TOOLS,
            tool_executor=self._execute_tool,
        )
        return collected

    def analyze(self, raw_data: dict[str, Any]) -> AgentSubReport:
        """Produce the structured financial sub-report from gathered data.

        Args:
            raw_data: Output of gather_data() — dict of tool results.

        Returns:
            AgentSubReport with financial findings, sourced claims, and confidence scores.
        """
        return self._produce_structured_report(FINANCIAL_ANALYZE_SYSTEM, raw_data)
