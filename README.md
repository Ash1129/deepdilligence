# DeepDiligence

DeepDiligence is a multi-agent investment due diligence system built for the Cornell ENMGT5400 final project. Given a public company, it runs four specialist AI agents in parallel, synthesizes their structured findings into an investment memo, and can generate weekly BUY/HOLD/SELL-style rankings across a stock watchlist.

## What It Does

- Runs specialist agents for financial analysis, team and culture, market competition, and risk sentiment.
- Pulls data from SEC EDGAR, NewsAPI, company websites, careers pages, and general web scraping.
- Produces typed Pydantic outputs instead of free-form agent text.
- Preserves claim-to-source traceability through source IDs and confidence scores.
- Surfaces cross-agent contradictions explicitly instead of hiding them.
- Supports a Streamlit UI for single-company analysis and weekly stock rankings.
- Caches API and LLM outputs locally to reduce cost during development.

## Architecture

```text
app.py
  ├─ Single Company page
  │   └─ Orchestrator
  │       ├─ FinancialAgent
  │       ├─ TeamAgent
  │       ├─ MarketAgent
  │       ├─ RiskAgent
  │       └─ SynthesisAgent → InvestmentMemo
  │
  └─ Weekly Rankings page
      ├─ build_watchlist()
      ├─ run_batch()
      └─ RecommendationEngine → WeeklyReport
```

The orchestrator runs the four specialist agents concurrently with `ThreadPoolExecutor`. Synthesis starts only after all specialist reports finish or fail gracefully. The Streamlit UI is intentionally synchronous: worker-thread callbacks never call `st.*`, while main-thread callbacks update `st.status()`.

## Key Files

- `app.py` — Streamlit app with Single Company and Weekly Rankings views.
- `src/orchestrator.py` — parallel specialist execution and synthesis coordination.
- `src/agents/` — specialist agents and shared ReAct-style base class.
- `src/data/` — SEC EDGAR, NewsAPI, careers-page, and web-scraping clients.
- `src/models/schemas.py` — Pydantic v2 data contracts.
- `src/models/prompts.py` — all agent and synthesis prompts.
- `src/scheduler/` — batch memo runner, stock screener, and recommendation engine.
- `src/evaluation/` — benchmark coverage and faithfulness scoring.
- `tests/` — unit tests for schemas, data clients, agents, synthesis, and evaluation.

## Setup

Use Python 3.11 or 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```bash
OPENAI_API_KEY=...
NEWS_API_KEY=...
SEC_EDGAR_USER_AGENT="DeepDiligence your-email@example.com"
```

The SEC EDGAR API requires a descriptive User-Agent with contact information.

## Run The App

```bash
streamlit run app.py
```

The app has two modes:

- **Single Company**: enter a company and optional ticker, then generate a due diligence memo.
- **Weekly Rankings**: select investment styles, sectors, stock count, and optional live S&P 500 screening.

## Run From CLI

Single-company example:

```bash
python3.11 run_pipeline.py
```

Weekly report:

```bash
python3.11 run_weekly.py --dry-run
python3.11 run_weekly.py --refresh
```

`--dry-run` uses cached memos only. `--refresh` reruns live analysis and may use OpenAI, NewsAPI, SEC EDGAR, and web requests.

## Testing

```bash
python3.11 -m pytest -q
```

Current local status: 54 passing tests.

## Data And Caching

DeepDiligence caches expensive calls under `data/cache/`:

- `data/cache/agents/` — structured specialist reports.
- `data/cache/memos/` — synthesized company memos.
- `data/cache/news/`, `data/cache/edgar/`, `data/cache/web/`, `data/cache/jobs/` — data-source responses.

These caches are intentionally excluded from git because they can be large, stale, and derived from API calls.

## Limitations

- Private companies have limited financial data because SEC EDGAR coverage is unavailable.
- The live screener depends on `yfinance`; if it fails, the app falls back to the curated universe.
- Agent caches are keyed by agent name, company, and ticker, not prompt or model version.
- The final memo preserves source IDs, but full source objects primarily live in specialist subreports and cache artifacts.
- This is a research/course project, not investment advice.

## Recommended Next Improvements

- Add model and prompt versioning to cache keys.
- Embed final memo source registries for self-contained JSON exports.
- Add PDF export from memo Markdown using WeasyPrint.
- Expand benchmark profiles and improve evaluation beyond keyword coverage.
- Add CI after the GitHub repository is created.
