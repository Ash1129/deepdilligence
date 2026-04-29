# DeepDiligence: Multi-Agent Due Diligence System

## Project Overview
DeepDiligence is a multi-agent AI system that automates investment due diligence. Given a company name, it deploys four specialized AI agents (Financial Analyst, Team & Culture, Market & Competitive, Risk & Sentiment), each researching a distinct dimension using public data. A synthesis agent reconciles their findings, detects contradictions, and produces a structured investment memo with confidence scores and source traceability.

## Tech Stack
- **Language**: Python 3.11+
- **Agent Framework**: CrewAI (start here) or LangGraph (migrate if needed)
- **LLM**: Codex API (anthropic SDK) — use Codex-sonnet-4-20250514 for agents, Codex-sonnet-4-20250514 for synthesis
- **Vector Store**: ChromaDB (local, pip install)
- **Web UI**: Streamlit
- **Data Sources**: SEC EDGAR (free API), NewsAPI (free tier), web scraping (BeautifulSoup/requests)
- **Output**: Structured Markdown → PDF (WeasyPrint)

## Project Structure
```
deepdiligence/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .env                    # API keys (ANTHROPIC_API_KEY, NEWS_API_KEY)
├── app.py                  # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── orchestrator.py     # Main orchestrator that dispatches to agents
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py   # Abstract base class for all agents
│   │   ├── financial.py    # Financial Analyst agent (SEC EDGAR, revenue signals)
│   │   ├── team.py         # Team & Culture agent (job postings, leadership)
│   │   ├── market.py       # Market & Competitive agent (competitors, positioning)
│   │   ├── risk.py         # Risk & Sentiment agent (news sentiment, litigation)
│   │   └── synthesis.py    # Synthesis agent (conflict detection, confidence scoring)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── edgar.py        # SEC EDGAR API client
│   │   ├── news.py         # News API client
│   │   ├── jobs.py         # Job posting scraper
│   │   └── web.py          # General web scraper (company sites, pricing pages)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py      # Pydantic models for agent outputs, memos, sources
│   │   └── prompts.py      # All agent system prompts and templates
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py    # Benchmark companies with known outcomes
│   │   ├── faithfulness.py # Claim-to-source traceability scoring
│   │   └── metrics.py      # Evaluation metrics and reporting
│   └── utils/
│       ├── __init__.py
│       ├── cache.py        # Response caching to save API costs during dev
│       └── config.py       # Configuration management
├── tests/
│   ├── test_agents.py
│   ├── test_data.py
│   └── test_synthesis.py
└── data/
    ├── cache/              # Cached API responses
    └── benchmarks/         # Benchmark company data (JSON)
```

## Architecture Principles
1. **Every agent produces a structured Pydantic output** — not free-text. Each sub-report has: findings (list of claims), sources (list of source objects with URL, retrieved_at, snippet), confidence_score (float 0-1), and conflicts (list of internal contradictions the agent noticed).
2. **The synthesis agent never sees raw data** — only structured sub-reports from specialist agents. It operates at the abstraction level of claims and evidence, not documents.
3. **Conflict detection is explicit** — when Agent A says revenue is growing but Agent D flags financial risk, the synthesis agent surfaces both with evidence and lets the reader decide. It does NOT silently pick a side.
4. **Source traceability is mandatory** — every claim in the final memo must link to at least one source. Unsourceable claims get flagged, not included.
5. **Cache everything during development** — save all API responses (LLM and data) to disk so rebuilds don't cost money.

## Key Design Decisions
- Use Pydantic for all inter-agent data contracts (strict typing, validation)
- Each agent has its own set of tools (web search, SEC fetch, etc.) and cannot access other agents' tools
- The orchestrator runs agents in parallel where possible (financial + team + market + risk can all run concurrently)
- Synthesis runs only after all specialist agents complete
- Streamlit UI shows real-time progress (which agent is running, intermediate results)

## Coding Standards
- Type hints on all functions
- Docstrings on all public methods
- Use `logging` module, not print statements
- Use `python-dotenv` for env management
- All agent prompts live in `src/models/prompts.py`, not inline
- Write tests for data pipeline functions and schema validation

## Current Phase
Phase 1: Architecture & Data Pipeline

## Important Notes
- SEC EDGAR requires a User-Agent header with contact email
- Cache all API responses locally during development to minimize costs
- NewsAPI free tier: 100 requests/day — enough for dev but cache aggressively
- Skip Glassdoor (API is closed) — use job posting volume as culture proxy
- For private companies, the Financial agent will have limited data — this is a known limitation, not a bug
