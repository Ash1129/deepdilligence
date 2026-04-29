# DeepDiligence Technical Report

## Project Summary

DeepDiligence is a multi-agent AI system for automated public-company investment due diligence. The system accepts a company name and optional stock ticker, runs four specialist research agents in parallel, reconciles their outputs with a synthesis agent, and presents a structured investment memo with confidence scores and source traceability. A second workflow generates weekly ranked recommendations across a selected stock universe.

The project is built as a Cornell ENMGT5400 final project and is implemented in Python with a Streamlit frontend, OpenAI-based LLM calls, Pydantic v2 schemas, SEC EDGAR, NewsAPI, BeautifulSoup-based scraping, and optional yfinance market screening.

## Problem And Motivation

Traditional investment diligence requires analysts to collect financial disclosures, read market and risk news, inspect leadership and hiring signals, and reconcile conflicting evidence. That process is slow and often inconsistent across companies. DeepDiligence explores whether a structured multi-agent architecture can make the diligence process faster, more repeatable, and more auditable.

The main design goal is not simply to generate a polished memo. The system is designed to preserve intermediate evidence, expose uncertainty, and keep each agent responsible for a distinct analytical dimension.

## System Architecture

The system has two primary workflows.

### Single-Company Due Diligence

The Streamlit app calls `Orchestrator`, which starts four specialist agents concurrently:

- `FinancialAgent`: SEC EDGAR filings, revenue data, investor-relations pages, and financial disclosures.
- `TeamAgent`: careers pages, leadership/team pages, and hiring posture.
- `MarketAgent`: company news, competitive landscape, product positioning, and partnerships.
- `RiskAgent`: lawsuits, investigations, layoffs, controversies, breaches, regulatory actions, and reputational risk.

Each specialist returns an `AgentSubReport` containing findings, sources, confidence score, internal conflicts, and a raw data summary. The `SynthesisAgent` receives only those structured subreports, not raw scraped documents. It produces an `InvestmentMemo` with an executive summary, section-level analysis, key sourced claims, surfaced conflicts, and overall confidence.

### Weekly Recommendation Workflow

The weekly workflow uses the scheduler modules:

- `screener.py` filters the curated stock universe or screens live S&P 500 movers with yfinance.
- `batch_runner.py` runs or loads cached memos for selected companies.
- `recommender.py` compares memos and produces a `WeeklyReport` with ranked ratings from `STRONG BUY` to `STRONG SELL`.

The recommendation engine acts as a portfolio-manager layer above the diligence agents.

## Data Contracts

Pydantic models in `src/models/schemas.py` define the core contracts:

- `Source`
- `AgentClaim`
- `ConflictingClaim`
- `AgentSubReport`
- `SynthesizedSection`
- `InvestmentMemo`

The weekly recommender defines additional Pydantic models:

- `StockRating`
- `WeeklyReport`

Strict schemas make outputs easier to validate, test, display, cache, and evaluate.

## Data Sources

The system integrates several public data sources:

- SEC EDGAR company tickers, submissions, and XBRL company facts.
- NewsAPI recent article search.
- Company websites and investor-relations pages via requests and BeautifulSoup.
- Careers pages as a proxy for hiring health and organizational posture.
- yfinance for optional live price and volume screening.

All expensive or rate-limited data calls are cached under `data/cache/`.

## Agent Design

Specialist agents inherit from `ReactAgent`, which adds two shared capabilities:

1. A ReAct-style loop where the model chooses tools, observes results, and continues until enough evidence is gathered.
2. A structured report step where the model is forced to call a function tool that maps to `AgentSubReport`.

This keeps the research process adaptive while preserving a predictable output contract.

## Synthesis Design

The synthesis agent intentionally operates only on specialist subreports. This reinforces separation of concerns:

- Specialists collect and interpret raw source material.
- Synthesis reconciles claims, source IDs, confidence scores, and conflicts.

The synthesis prompt requires explicit conflict detection, source preservation, deduplication, confidence calibration, and a balanced executive summary.

## Streamlit Implementation

The UI is implemented in `app.py` with two pages:

- Single Company
- Weekly Rankings

The single-company pipeline uses `st.status()` for progress updates. Specialist agents run in background worker threads inside the orchestrator, but Streamlit UI calls are kept on the main thread. This avoids the common Streamlit failure mode where worker threads attempt to call `st.*` without a script-run context.

## Evaluation

The project includes a lightweight evaluation framework:

- `faithfulness.py` scores source traceability by measuring how many memo claims include at least one source ID.
- `metrics.py` compares a memo against benchmark company profiles using keyword-based fact, risk, and strength coverage.
- Benchmark JSON files live in `data/benchmarks/`.

This is intentionally simple but useful for course-level validation and regression checks.

## Current Validation

The current local test suite passes:

```text
54 passed
```

The tests cover:

- Pydantic schema validation.
- Cache behavior.
- SEC EDGAR client behavior with mocked responses.
- NewsAPI client behavior with mocked responses.
- Web and careers-page scraping.
- Agent report construction.
- Synthesis memo construction and fallback JSON extraction.
- Faithfulness and benchmark evaluation metrics.

## Current Artifacts

The project contains cached memo outputs for many public companies under `data/cache/memos/`, including Apple, Microsoft, NVIDIA, Tesla, Amazon, Alphabet, Meta, AMD, JPMorgan, Netflix, Salesforce, and others. A weekly report artifact exists under `data/recommendations/`.

These generated artifacts are useful for demos but are excluded from git by default because they are derived outputs and may become stale.

## Limitations

- Private-company financial analysis is inherently limited because SEC EDGAR data is unavailable.
- Web scraping is best-effort and can fail on JavaScript-heavy pages or sites that block automated requests.
- NewsAPI free-tier limits require aggressive caching.
- Current cache keys do not include model name, prompt version, or tool schema version.
- The final memo stores claim source IDs but does not embed a full source registry in the exported memo JSON.
- The recommendation engine is LLM-based and should be treated as an analytical aid, not investment advice.

## Future Work

- Add prompt/model versioning to cache keys.
- Add a self-contained source registry to `InvestmentMemo`.
- Add PDF export for investment memos.
- Add GitHub Actions CI after repository publication.
- Expand benchmark datasets and add semantic evaluation.
- Add retry/backoff handling for NewsAPI, EDGAR, and OpenAI calls.
- Improve URL discovery for careers, investor-relations, and leadership pages.

## Conclusion

DeepDiligence demonstrates a practical, auditable multi-agent pattern for investment research. The system separates specialist evidence gathering from synthesis, uses strict schemas for inter-agent contracts, caches expensive operations, and provides both single-company memo generation and portfolio-style weekly ranking workflows. It is suitable as a working course project prototype and a foundation for further experimentation with structured agentic diligence systems.
