"""Abstract base class for all specialist agents."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.models.schemas import AgentSubReport

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class that all specialist agents must implement.

    Each agent receives a company name (and optional context), runs its
    research pipeline, and returns a structured AgentSubReport.
    """

    def __init__(self, company_name: str, ticker: str | None = None, **kwargs: Any):
        self.company_name = company_name
        self.ticker = ticker
        self.logger = logging.getLogger(f"{__name__}.{self.agent_name}")

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique identifier for this agent (e.g., 'financial_analyst')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this agent investigates."""
        ...

    @abstractmethod
    def gather_data(self) -> dict[str, Any]:
        """Collect raw data from external sources.

        Returns a dict of data keyed by source name. Each agent implements
        this with its own set of data tools (EDGAR, NewsAPI, web scraper, etc.).
        """
        ...

    @abstractmethod
    def analyze(self, raw_data: dict[str, Any]) -> AgentSubReport:
        """Analyze gathered data and produce a structured sub-report.

        This is where the agent calls the LLM to interpret raw data,
        extract claims, assign confidence scores, and detect conflicts.

        Args:
            raw_data: Output from gather_data().

        Returns:
            A fully populated AgentSubReport.
        """
        ...

    def run(self) -> AgentSubReport:
        """Execute the full agent pipeline: gather data, then analyze.

        This is the main entry point called by the orchestrator.
        """
        self.logger.info("Starting %s for %s", self.agent_name, self.company_name)

        self.logger.info("Gathering data...")
        raw_data = self.gather_data()

        self.logger.info("Analyzing data (%d sources)...", len(raw_data))
        report = self.analyze(raw_data)

        self.logger.info(
            "Completed %s: %d findings, confidence=%.2f",
            self.agent_name,
            len(report.findings),
            report.confidence_score,
        )
        return report
