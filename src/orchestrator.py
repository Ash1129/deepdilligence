"""Main orchestrator: runs specialist agents in parallel, then synthesizes findings."""

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.agents.financial import FinancialAgent
from src.agents.market import MarketAgent
from src.agents.risk import RiskAgent
from src.agents.synthesis import SynthesisAgent
from src.agents.team import TeamAgent
from src.models.schemas import AgentSubReport, InvestmentMemo
from src.utils.company_validation import validate_company_identity

logger = logging.getLogger(__name__)

# Type aliases for progress callbacks
OnAgentStart = Callable[[str], None]       # agent_name → None
OnAgentComplete = Callable[[str, AgentSubReport], None]  # agent_name, report → None
OnSynthesisStart = Callable[[], None]


class Orchestrator:
    """Coordinates the full due diligence pipeline for a given company.

    Execution flow:
    1. Spin up all four specialist agents in parallel (ThreadPoolExecutor)
    2. Collect their AgentSubReport outputs (with graceful error handling per agent)
    3. Pass all sub-reports to SynthesisAgent
    4. Return the final InvestmentMemo

    Progress callbacks let callers (e.g. Streamlit) stream real-time status updates
    without polling.

    Example (sync):
        memo = Orchestrator("Apple Inc", ticker="AAPL").run()

    Example (async, e.g. inside an async framework):
        memo = await Orchestrator("Apple Inc", ticker="AAPL").run_async()

    Example (with progress callbacks for Streamlit):
        orch = Orchestrator(
            "Apple Inc",
            ticker="AAPL",
            on_agent_start=lambda name: st.write(f"▶ {name} running..."),
            on_agent_complete=lambda name, _: st.write(f"✓ {name} done"),
            on_synthesis_start=lambda: st.write("⚙ Synthesizing..."),
        )
        memo = orch.run()
    """

    def __init__(
        self,
        company_name: str,
        ticker: str | None = None,
        on_agent_start: OnAgentStart | None = None,
        on_agent_complete: OnAgentComplete | None = None,
        on_synthesis_start: OnSynthesisStart | None = None,
    ) -> None:
        self.company_name = company_name
        self.ticker = ticker

        # Progress callbacks (default to no-ops)
        self._on_agent_start = on_agent_start or (lambda name: None)
        self._on_agent_complete = on_agent_complete or (lambda name, report: None)
        self._on_synthesis_start = on_synthesis_start or (lambda: None)

        self._synthesis: SynthesisAgent | None = None

    # ─── Sync entry point ────────────────────────────────────────────────────

    def run(self) -> InvestmentMemo:
        """Run the full pipeline synchronously and return the InvestmentMemo.

        Specialist agents run in parallel via ThreadPoolExecutor; synthesis runs
        after all four complete (or fail gracefully). Safe to call from any context
        including Streamlit (no asyncio event loop required).
        """
        start = time.perf_counter()
        identity = validate_company_identity(self.company_name, self.ticker)
        self.company_name = identity.company_name
        self.ticker = identity.ticker
        self._synthesis = SynthesisAgent(self.company_name)

        logger.info("Orchestrator starting for %s (ticker=%s)", self.company_name, self.ticker)

        sub_reports = self._run_specialists_parallel()

        self._on_synthesis_start()
        logger.info("All specialists done — starting synthesis")

        memo = self._synthesis.synthesize(
            sub_reports,
            extra_metadata={"elapsed_seconds": round(time.perf_counter() - start, 1)},
        )

        elapsed = round(time.perf_counter() - start, 1)
        logger.info("Pipeline complete in %.1fs — overall_confidence=%.2f", elapsed, memo.overall_confidence)
        return memo

    # ─── Async entry point ───────────────────────────────────────────────────

    async def run_async(self) -> InvestmentMemo:
        """Async version of run() for use inside async frameworks.

        Delegates blocking work to a thread pool so the event loop stays free.
        """
        return await asyncio.to_thread(self.run)

    # ─── Parallel specialist execution ───────────────────────────────────────

    def _run_specialists_parallel(self) -> list[AgentSubReport]:
        """Run all four specialist agents concurrently in a thread pool.

        Each agent is independent — failures are caught per-agent so one bad
        agent never blocks the others. A failed agent produces a minimal error
        sub-report (zero findings, zero confidence) so synthesis always receives
        exactly four inputs.

        Returns:
            List of four AgentSubReport objects (one per specialist, in order:
            financial, team, market, risk).
        """
        agents = [
            FinancialAgent(self.company_name, self.ticker),
            TeamAgent(self.company_name, self.ticker),
            MarketAgent(self.company_name, self.ticker),
            RiskAgent(self.company_name, self.ticker),
        ]

        results: dict[str, AgentSubReport] = {}

        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent") as pool:
            future_to_agent = {
                pool.submit(self._run_one_agent, agent): agent for agent in agents
            }

            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    report = future.result()
                    results[agent.agent_name] = report
                    self._on_agent_complete(agent.agent_name, report)
                    logger.info(
                        "Agent %s complete — %d findings, confidence=%.2f",
                        agent.agent_name,
                        len(report.findings),
                        report.confidence_score,
                    )
                except Exception as exc:
                    logger.error(
                        "Agent %s failed with %s: %s",
                        agent.agent_name, type(exc).__name__, exc,
                        exc_info=True,
                    )
                    error_report = self._error_subreport(agent.agent_name, exc)
                    results[agent.agent_name] = error_report
                    self._on_agent_complete(agent.agent_name, error_report)

        # Return in canonical order for consistent memo section ordering
        ordered = ["financial_analyst", "team_culture", "market_competitive", "risk_sentiment"]
        return [results[name] for name in ordered if name in results]

    def _run_one_agent(self, agent: Any) -> AgentSubReport:
        """Run a single agent, firing the on_agent_start callback first.

        The callback is wrapped in try/except because this method runs inside
        a ThreadPoolExecutor worker thread.  UI callbacks (e.g. Streamlit's
        st.status.write) require the main-thread session context and will raise
        if called from a worker thread.  A failed callback must never prevent
        the agent itself from running.
        """
        try:
            self._on_agent_start(agent.agent_name)
        except Exception as cb_exc:
            logger.debug("on_agent_start callback raised (safe to ignore): %s", cb_exc)
        logger.info("Starting agent: %s", agent.agent_name)
        return agent.run()

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _error_subreport(agent_name: str, exc: Exception) -> AgentSubReport:
        """Build a zero-findings sub-report when an agent crashes.

        The synthesis agent will see this and can note data gaps in the memo.
        """
        return AgentSubReport(
            agent_name=agent_name,
            findings=[],
            sources=[],
            confidence_score=0.0,
            conflicts=[],
            raw_data_summary=(
                f"Agent failed with {type(exc).__name__}: {exc}. "
                "No data was gathered for this dimension."
            ),
        )
