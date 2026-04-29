# DeepDiligence

DeepDiligence is a multi-agent investment due diligence system built for the Cornell ENMGT5400 final project. Given a public company, it runs four specialist AI agents in parallel, synthesizes their structured findings into a sourced investment memo, and generates weekly BUY/HOLD/SELL-ranked recommendations across a stock watchlist.

---

## What It Does

- Runs four specialist agents concurrently: **Financial**, **Team & Culture**, **Market & Competitive**, **Risk & Sentiment**
- Pulls live data from SEC EDGAR, NewsAPI, company websites, careers pages, and web scraping
- Produces strict Pydantic-typed outputs — no free-form agent text leaks through
- Every claim links to at least one source ID; unsourced claims are flagged, not silently included
- Surfaces cross-agent contradictions explicitly rather than resolving them silently
- Scores each memo with faithfulness (source traceability) and benchmark coverage metrics
- Weekly workflow: screen S&P 500 top movers → run batch diligence → generate ranked report

---

## Architecture

```text
app.py
  ├─ Single Company page
  │   └─ Orchestrator  (ThreadPoolExecutor — 4 agents in parallel)
  │       ├─ FinancialAgent    → SEC EDGAR, investor-relations pages
  │       ├─ TeamAgent         → careers pages, leadership signals
  │       ├─ MarketAgent       → competitors, positioning, news
  │       ├─ RiskAgent         → litigation, regulatory, reputational risk
  │       └─ SynthesisAgent    → InvestmentMemo (sourced claims, conflicts, confidence)
  │
  └─ Weekly Rankings page
      ├─ build_watchlist()     → curated universe or live S&P 500 screener (yfinance)
      ├─ batch_runner.py       → per-company memo pipeline with disk cache
      └─ RecommendationEngine  → WeeklyReport (STRONG BUY → STRONG SELL)
```

Streamlit runs the pipeline synchronously on the main thread via `st.status()`. Worker threads inside the orchestrator never call `st.*` — only main-thread callbacks do.

---

## Key Files

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit app — Single Company and Weekly Rankings pages |
| `src/orchestrator.py` | Parallel specialist dispatch + synthesis coordination |
| `src/agents/` | `financial.py`, `team.py`, `market.py`, `risk.py`, `synthesis.py` |
| `src/data/` | SEC EDGAR, NewsAPI, web scraper, careers-page scraper |
| `src/models/schemas.py` | Pydantic v2 contracts for all inter-agent data |
| `src/models/prompts.py` | All agent system prompts and tool definitions |
| `src/scheduler/screener.py` | Curated universe filter + live S&P 500 screener |
| `src/scheduler/batch_runner.py` | Batch memo runner with per-ticker disk cache |
| `src/scheduler/recommender.py` | Portfolio-manager LLM → ranked `WeeklyReport` |
| `src/evaluation/faithfulness.py` | Source-traceability scoring (A–F grade) |
| `src/evaluation/metrics.py` | Benchmark fact/risk coverage + composite score |
| `data/stock_universe.json` | 95 stocks tagged by sector and investment style |
| `data/benchmarks/` | 10 benchmark company profiles for evaluation |
| `run_weekly.py` | CLI for weekly batch runs |
| `tests/` | 54 unit tests |

---

## Setup

Requires Python 3.11 or 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

`.env` keys required:

```
OPENAI_API_KEY=sk-...
NEWS_API_KEY=...
SEC_EDGAR_USER_AGENT="DeepDiligence your-email@example.com"
```

The SEC EDGAR API requires a descriptive User-Agent string with a contact email.

---

## Run The App

```bash
streamlit run app.py
```

**Single Company** — enter a company name and optional ticker, click Run Analysis. The memo appears with tabbed sections (Executive Summary, Financial, Team, Market, Risk, Stats & Sources, Export).

**Weekly Rankings** — choose investment styles, sectors, stock count, and whether to use the live S&P 500 screener or the curated universe. Click Generate Weekly Report.

---

## Run From CLI

```bash
# Single company
python3.11 run_pipeline.py

# Weekly report — uses cached memos only (fast, no API calls)
python3.11 run_weekly.py --dry-run

# Weekly report — reruns all pipelines live
python3.11 run_weekly.py --refresh
```

---

## Testing

```bash
python3.11 -m pytest -q
```

```
54 passed in 1.2s
```

Tests cover: Pydantic schema validation, disk cache, SEC EDGAR client, NewsAPI client, web and careers-page scraping, agent report construction, synthesis memo building, faithfulness scoring, and benchmark coverage metrics.

---

## Evaluation Results

Scores across ten benchmarked companies (memos generated with `gpt-5.4-mini`):

| Ticker | Company | Confidence | Faithfulness | Facts | Risks | Composite | Grade |
|--------|---------|-----------|-------------|-------|-------|-----------|-------|
| AAPL | Apple Inc | 78% | 100% A | 62% | 83% | 88% | **A** |
| MSFT | Microsoft | 84% | 80% B | 100% | 100% | 87% | **A** |
| JPM | JPMorgan Chase | 76% | 100% A | 75% | 67% | 83% | **B** |
| CRM | Salesforce | 72% | 83% B | 88% | 67% | 79% | **B** |
| NFLX | Netflix | 73% | 81% B | 88% | 50% | 76% | **B** |
| NVDA | NVIDIA | 78% | — | 75% | 67% | 46% | D |
| AMZN | Amazon | 74% | — | 50% | 83% | 44% | D |
| META | Meta Platforms | 72% | — | 75% | 67% | 44% | D |
| GOOGL | Alphabet | 76% | — | 75% | 67% | 43% | D |
| TSLA | Tesla | 60% | — | 38% | 67% | 33% | F |

- **Confidence**: overall memo confidence as assessed by the synthesis agent
- **Faithfulness**: share of claims with ≥1 source ID linked (A = ≥90%)
- **Facts / Risks**: keyword coverage against benchmark known-facts / known-risks lists
- **Composite**: 40% fact coverage + 40% faithfulness + 20% confidence

Lower faithfulness scores reflect runs where the synthesis agent did not carry source IDs through from specialist subreports — an identified improvement area.

---

## Sample Weekly Report (2026-W18, 7 Technology stocks)

```
#1  NVDA  STRONG BUY   28%  — AI infrastructure leader, 62% YoY revenue growth
#2  NOW   BUY          18%  — High-quality SaaS, durable enterprise moat
#3  SNOW  BUY          15%  — Cloud data platform, strong net retention
#4  TXN   HOLD          —   — Mature semiconductor, limited near-term catalyst
#5  ORCL  HOLD          —   — Stable but evidence-constrained
#6  AMD   SELL          —   — Competitive pressure, evidence gaps
#7  PLTR  STRONG SELL   —   — Weak diligence packet, unverifiable traction claims
```

---

## Data and Caching

All expensive calls are cached under `data/cache/`:

| Path | Contents |
|------|---------|
| `data/cache/agents/` | Per-agent structured subreports |
| `data/cache/memos/` | Full synthesized memos per ticker |
| `data/cache/news/` | NewsAPI responses |
| `data/cache/edgar/` | SEC EDGAR API responses |
| `data/cache/web/` | Scraped web pages |
| `data/cache/jobs/` | Careers-page scrape results |

Cache keys are SHA-256 hashes of `(agent_name, company, ticker)`. Cache files are excluded from git.

---

## Limitations

- Private-company financial data is unavailable (no SEC EDGAR coverage)
- Web scraping is best-effort and fails on JS-heavy or bot-blocking pages
- NewsAPI free tier (100 req/day) requires aggressive caching; 429 errors appear in some memos
- Cache keys do not include model version or prompt version
- The recommendation engine is LLM-based — treat output as an analytical aid, not investment advice

---

## Potential Improvements

- Add prompt and model versioning to cache keys
- Embed a full source registry in exported `InvestmentMemo` JSON
- Add PDF export using WeasyPrint
- Add GitHub Actions CI with test and lint steps
- Expand benchmark profiles; add semantic (embedding-based) coverage scoring
- Add retry and backoff for NewsAPI, EDGAR, and OpenAI rate limits
