"""Team & Culture agent — hiring velocity, leadership quality, and culture signals."""

import logging
from typing import Any

from src.agents.react_agent import ReactAgent
from src.data.jobs import scrape_careers_page
from src.data.web import scrape_url
from src.models.prompts import TEAM_ANALYZE_SYSTEM, TEAM_REACT_SYSTEM
from src.models.schemas import AgentSubReport

logger = logging.getLogger(__name__)

# Tool definitions in OpenAI format
_TEAM_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "scrape_careers_page",
            "description": (
                "Scrape a company's careers or jobs page to extract open role count, "
                "job titles, and department distribution. "
                "Returns: job_count, job_titles (list), departments (list), raw_text. "
                "Try common URLs like https://company.com/careers or https://company.com/jobs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "careers_url": {
                        "type": "string",
                        "description": "Full URL to the company's careers/jobs page",
                    }
                },
                "required": ["careers_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": (
                "Fetch and extract text from any URL. "
                "Use for About pages, Team/Leadership pages, LinkedIn company pages, "
                "Glassdoor profiles, or any page with team/culture information."
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


class TeamAgent(ReactAgent):
    """Analyst for team composition, leadership quality, and hiring health.

    Data sources:
    - Company careers / jobs page (job count and department mix)
    - About / Team / Leadership pages (executive backgrounds)
    - LinkedIn company page (headcount signals)
    - General web scraping for culture and team news
    """

    @property
    def agent_name(self) -> str:
        return "team_culture"

    @property
    def description(self) -> str:
        return (
            "Analyzes team composition, leadership experience, hiring velocity, "
            "and culture signals from careers pages and public company profiles."
        )

    def _execute_tool(self, name: str, tool_input: dict) -> Any:
        """Route a Claude tool call to the appropriate data function.

        Args:
            name: Tool name as defined in _TEAM_TOOLS.
            tool_input: Parameters dict from the Claude tool call.

        Returns:
            JSON-serializable result from the underlying data function.
        """
        if name == "scrape_careers_page":
            result = scrape_careers_page(tool_input["careers_url"])
            # Trim raw_text to keep context manageable
            if result.get("raw_text"):
                result = dict(result)
                result["raw_text"] = result["raw_text"][:3000]
            return result

        if name == "scrape_url":
            result = scrape_url(tool_input["url"])
            if result.get("text"):
                result = dict(result)
                result["text"] = result["text"][:5000]
            return result

        raise ValueError(f"TeamAgent: unknown tool '{name}'")

    def gather_data(self) -> dict[str, Any]:
        """Run the ReAct loop to gather team and culture data.

        The LLM constructs likely URLs for careers and about pages, scrapes them,
        and iterates until it has enough signal on team composition and hiring.
        """
        initial_message = (
            f"Conduct team and culture due diligence on: {self.company_name}\n\n"
            "Use the tools to gather:\n"
            "- Open job count and department breakdown from the careers page\n"
            "- Executive / leadership team names and backgrounds from the About or Team page\n"
            "- Any publicly visible culture signals (remote policy, stated values)\n\n"
            "Start by trying the careers page at likely URLs such as "
            f"https://www.{self.company_name.lower().replace(' ', '')}.com/careers "
            "or similar. Then scrape the About/Team page. "
            "Stop when you have characterized the team size, leadership, and hiring posture."
        )

        _, collected = self._run_react_loop(
            system_prompt=TEAM_REACT_SYSTEM,
            initial_user_message=initial_message,
            tools=_TEAM_TOOLS,
            tool_executor=self._execute_tool,
        )
        return collected

    def analyze(self, raw_data: dict[str, Any]) -> AgentSubReport:
        """Produce the structured team & culture sub-report from gathered data.

        Args:
            raw_data: Output of gather_data() — dict of tool results.

        Returns:
            AgentSubReport with team findings, sourced claims, and confidence scores.
        """
        return self._produce_structured_report(TEAM_ANALYZE_SYSTEM, raw_data)
