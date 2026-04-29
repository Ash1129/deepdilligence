# DeepDiligence — Technical Report

**Cornell ENMGT5400 — Applications of AI for Engineering Managers**
**Final Project | April 2026**

---

## 1. Problem and Motivation

Investment due diligence is a time-intensive, analyst-driven process that involves reading SEC filings, monitoring news and risk signals, evaluating competitive positioning, and assessing leadership and hiring health — across multiple companies simultaneously. The process is slow, expensive, and inconsistent across analysts. It also produces outputs that are hard to audit: a traditional analyst memo does not show which claim came from which source, where two data sources disagreed, or how confident the analyst was in each finding.

DeepDiligence explores whether a structured multi-agent AI architecture can automate a meaningful portion of this process while remaining auditable. The design goal is not to produce a polished narrative. It is to preserve intermediate evidence, expose uncertainty, surface contradictions between sources, and produce a typed output that can be evaluated, tested, and compared across companies.

---

## 2. System Architecture

The system has two primary workflows.

### 2.1 Single-Company Due Diligence

```
Orchestrator (ThreadPoolExecutor)
  ├─ FinancialAgent
  ├─ TeamAgent
  ├─ MarketAgent
  └─ RiskAgent
         ↓  (all complete)
  SynthesisAgent  →  InvestmentMemo
```

The `Orchestrator` launches four specialist agents concurrently. Each returns an `AgentSubReport` containing findings, sources, a confidence score, and internal conflicts the agent noticed. Once all four finish (or fail gracefully), the `SynthesisAgent` receives only the structured subreports — not raw scraped text — and produces an `InvestmentMemo`.

### 2.2 Weekly Recommendation Workflow

```
build_watchlist()       →  selected stocks (filtered or S&P 500 screener)
batch_runner.run_batch()  →  dict[ticker, InvestmentMemo]  (cache-first)
RecommendationEngine.generate()  →  WeeklyReport  (STRONG BUY → STRONG SELL)
```

The weekly workflow takes user investment preferences (style, sector, count), builds a watchlist from a 95-stock curated universe or from live S&P 500 price/volume data via yfinance, runs or loads cached memos, and feeds them to a portfolio-manager LLM layer that produces ranked ratings.

---

## 3. Agent Design

### 3.1 Specialist Agents (ReAct Loop)

Each specialist agent inherits from `ReactAgent`, which implements a multi-turn tool-use loop:

1. The model receives a system prompt defining its analytical dimension and available tools.
2. It calls tools (SEC EDGAR lookup, news search, web scrape, careers scrape) and observes results.
3. It continues calling tools until it has enough evidence or hits an iteration limit.
4. It is then forced (via OpenAI tool-choice) to call a structured `produce_report` function that maps to `AgentSubReport`.

This keeps research adaptive while guaranteeing a predictable output contract.

**Tools available per agent:**

| Agent | Tools |
|-------|-------|
| FinancialAgent | SEC EDGAR CIK lookup, SEC filings list, revenue data, web scrape (investor-relations) |
| TeamAgent | Careers-page scrape, web scrape (leadership/about pages) |
| MarketAgent | NewsAPI search, web scrape (product/pricing pages) |
| RiskAgent | NewsAPI search (adverse terms), web scrape (SEC risk filings) |

### 3.2 Synthesis Agent

The synthesis agent intentionally receives only the four `AgentSubReport` objects — not raw data. This enforces a clean separation between evidence collection (specialist layer) and reconciliation (synthesis layer). The synthesis prompt requires:

- Source-ID preservation from subreports to final claims
- Explicit conflict detection and surfacing (not silent resolution)
- Per-section confidence scoring
- A balanced executive summary naming both bull and bear cases

### 3.3 Recommendation Engine

The recommendation engine acts as a portfolio-manager layer above the memo pipeline. It receives formatted memo text for all analysed companies, compares them on financial health, competitive moat, team quality, and risk-adjusted return, and calls a forced tool to produce `StockRating` objects (rating, rank, bull/bear case, suggested weight, confidence) and a `WeeklyReport`.

---

## 4. Data Contracts

All inter-agent data uses Pydantic v2 models defined in `src/models/schemas.py`:

```
Source            — url, source_type, retrieved_at, snippet, credibility_score
AgentClaim        — text, confidence, source_ids, reasoning
ConflictingClaim  — description, claim_a, claim_b, resolution
AgentSubReport    — agent_name, findings, sources, confidence_score, conflicts, raw_data_summary
SynthesizedSection — title, content, claims, conflicting_claims, confidence_score
InvestmentMemo    — company_name, executive_summary, sections, overall_confidence, metadata
```

Weekly recommender contracts:

```
StockRating   — ticker, rating, rank, bull_case, bear_case, rationale, weight, confidence
WeeklyReport  — week_of, ratings, top_picks, avoid, macro_commentary, sector_views
```

Strict typing makes outputs testable, cacheable, displayable, and evaluable without parsing free text.

---

## 5. Data Sources

| Source | Used By | Notes |
|--------|---------|-------|
| SEC EDGAR | FinancialAgent | CIK lookup, filings metadata, XBRL company facts (revenue). Free, no key required. |
| NewsAPI | MarketAgent, RiskAgent | Recent article search. Free tier: 100 req/day; aggressive caching required. |
| Web scraping | All agents | BeautifulSoup + requests; best-effort, fails on JS-heavy pages. |
| Careers pages | TeamAgent | Job-posting volume as hiring-health proxy. |
| yfinance | Screener | 5-day price/volume for S&P 500 top-movers ranking. Falls back to curated universe on failure. |

All data calls are cached under `data/cache/` using SHA-256 keys derived from request parameters. This kept API costs manageable across dozens of development iterations.

---

## 6. Streamlit UI Implementation

The frontend has two pages: **Single Company** and **Weekly Rankings**.

**Thread safety.** The Orchestrator's `ThreadPoolExecutor` runs specialist agents in background threads. A naive implementation would attempt to call `st.*` from those threads, which fails with a `NoSessionContext` error in Streamlit 1.34+. The solution:

- `on_agent_start` callback (fires from worker thread) → logging only, no Streamlit calls
- `on_agent_complete` and `on_synthesis_start` callbacks (fire from main thread's `as_completed` loop) → safe to call `status.write()`
- `st.status()` runs the entire pipeline synchronously on the main thread and shows live progress

**Weekly Rankings preferences.** Users can filter by investment style (Growth / Value / Dividend Income / Momentum), sector (all 11 GICS sectors), stock count (3–20), and toggle the live S&P 500 screener on or off. When the screener is on, stocks are ranked by price change, volume, or a combined score before being deep-dived.

---

## 7. Evaluation Framework

### 7.1 Faithfulness Scoring

`score_faithfulness(memo)` measures source traceability: what fraction of claims in the memo have at least one source ID linked.

| Grade | Threshold |
|-------|-----------|
| A | ≥ 90% |
| B | ≥ 75% |
| C | ≥ 60% |
| D | ≥ 40% |
| F | < 40% |

### 7.2 Benchmark Coverage

Ten benchmark profiles in `data/benchmarks/` each contain `known_facts`, `known_risks`, and `known_strengths` lists. `compute_metrics(memo, benchmark)` uses fuzzy keyword matching (≥50% significant words must appear in memo text) to measure fact and risk coverage.

**Composite score** = 40% fact coverage + 40% faithfulness + 20% overall confidence.

### 7.3 Results

Evaluation across ten benchmarked companies:

| Ticker | Company | Confidence | Faithfulness | Facts | Risks | Composite | Grade |
|--------|---------|-----------|-------------|-------|-------|-----------|-------|
| AAPL | Apple Inc | 78% | 100% (A) | 62% | 83% | **88%** | **A** |
| MSFT | Microsoft | 84% | 80% (B) | 100% | 100% | **87%** | **A** |
| JPM | JPMorgan Chase | 76% | 100% (A) | 75% | 67% | **83%** | **B** |
| CRM | Salesforce | 72% | 83% (B) | 88% | 67% | **79%** | **B** |
| NFLX | Netflix | 73% | 81% (B) | 88% | 50% | **76%** | **B** |
| NVDA | NVIDIA | 78% | 0% (F) | 75% | 67% | 46% | D |
| AMZN | Amazon | 74% | 0% (F) | 50% | 83% | 44% | D |
| META | Meta Platforms | 72% | 0% (F) | 75% | 67% | 44% | D |
| GOOGL | Alphabet | 76% | 0% (F) | 75% | 67% | 43% | D |
| TSLA | Tesla | 60% | 0% (F) | 38% | 67% | 33% | F |

**Observation on faithfulness variance.** Five companies score 0% faithfulness, meaning the synthesis agent did not carry source IDs from specialist subreports through to final memo claims. This appears correlated with runs where the synthesis model returned claims as plain text strings rather than structured `AgentClaim` objects with `source_ids`. The five companies with high faithfulness (AAPL, MSFT, JPM, CRM, NFLX) had runs where the forced-tool-call output was structured correctly. This is an identified reliability issue with the synthesis step, not a data collection failure — the source evidence exists in the subreports but is not always propagated.

**Finding count.** All ten memos produced 37–45 findings per company across 5 sections (Financial, Team & Culture, Market & Competitive, Risk & Sentiment, Synthesis).

**Source count.** Sources per memo ranged from 21 to 44.

**Cross-agent conflicts.** The synthesis agent detected 0–16 cross-agent conflicts per memo. JPMorgan (16 conflicts) and Apple (9) produced the most, reflecting genuine tension between financial signals from EDGAR and risk/news signals from NewsAPI.

---

## 8. Weekly Recommendation Sample

A representative weekly report generated on 2026-04-29 across 7 Technology stocks:

| Rank | Ticker | Rating | Weight | Rationale (summary) |
|------|--------|--------|--------|---------------------|
| 1 | NVDA | STRONG BUY | 28% | AI infrastructure leader, 62% YoY revenue growth, 73%+ gross margins |
| 2 | NOW | BUY | 18% | High-quality SaaS, durable enterprise moat, Agentforce AI platform |
| 3 | SNOW | BUY | 15% | Cloud data platform, strong net retention, AI integration momentum |
| 4 | TXN | HOLD | — | Mature semiconductor, strong dividend but limited near-term catalyst |
| 5 | ORCL | HOLD | — | Stable cloud growth but evidence-constrained diligence packet |
| 6 | AMD | SELL | — | Competitive pressure from NVDA, evidence gaps in market share data |
| 7 | PLTR | STRONG SELL | — | Weak diligence packet, unverifiable commercial traction claims |

Top picks: NVDA, NOW, SNOW. Avoid: AMD, PLTR.

---

## 9. Test Coverage

```
54 passed in 1.2s
```

Test modules and coverage:

| File | Tests | Areas |
|------|-------|-------|
| `test_agents.py` | 17 | `AgentSubReport` construction, confidence clamping, conflict parsing, cache keys, orchestrator instantiation |
| `test_data.py` | 20 | Pydantic schemas, disk cache, SEC EDGAR client (mocked), NewsAPI client (mocked), web scraper, careers scraper |
| `test_synthesis.py` | 17 | Faithfulness scoring (7 tests), benchmark coverage metrics (5 tests), synthesis memo building (5 tests) |

---

## 10. Design Decisions and Trade-offs

**Why OpenAI (gpt-5.4-mini) instead of Anthropic Claude?**
The project CLAUDE.md specified Claude, but during implementation the forced-tool-call pattern (needed for structured JSON outputs) was more reliable with the OpenAI tool-choice API. `gpt-5.4-mini` also provided faster iteration cycles during development.

**Why synchronous Streamlit instead of async/background threads?**
Early attempts to update `st.session_state` and `st.status()` from background threads caused `NoSessionContext` errors because Streamlit's script-run context is thread-local. The synchronous `st.status()` pattern was more reliable and simpler to reason about. The 4 specialist agents still run in parallel inside the orchestrator — only the UI layer is synchronous.

**Why Pydantic for inter-agent contracts?**
Free-text agent outputs are unparseable for downstream evaluation, caching, and display. Pydantic models provide validation at boundaries, clear error messages when an agent returns malformed output, and type-safe downstream consumption.

**Why keyword-based evaluation instead of embedding similarity?**
The benchmark evaluation uses fuzzy keyword matching as a fast, interpretable baseline. Embedding-based semantic matching would be more robust but adds a vector-store dependency and makes scores harder to explain to non-technical stakeholders. Keyword coverage is sufficient for course-level validation.

---

## 11. Limitations

- **Private companies**: SEC EDGAR data is unavailable for private companies. The Financial agent falls back to web scraping only.
- **NewsAPI rate limits**: The free tier allows 100 requests/day. Several cached memos contain 429 error observations from the Risk agent, reducing risk coverage scores.
- **Web scraping fragility**: JavaScript-heavy pages and bot-blocking return empty or partial results. Some company websites (e.g. Tesla) returned 403 on all career/IR page attempts.
- **Source ID propagation**: As noted in Section 7.3, the synthesis agent does not reliably carry source IDs through to final memo claims in all runs, reducing faithfulness scores.
- **Cache versioning**: Cache keys include agent name, company, and ticker but not model version or prompt version. Stale cache entries survive model changes.
- **Not investment advice**: All outputs are analytical aids for research purposes only.

---

## 12. Future Work

1. **Source-registry embedding**: attach a full `sources` dict to the exported `InvestmentMemo` JSON so memos are self-contained.
2. **Prompt and model versioning in cache keys**: invalidate stale caches automatically on prompt changes.
3. **Semantic evaluation**: replace keyword coverage with embedding-based claim matching against benchmark facts.
4. **PDF export**: WeasyPrint rendering from memo Markdown.
5. **GitHub Actions CI**: automated test and lint on push.
6. **Retry and backoff**: structured retry for NewsAPI, EDGAR, and OpenAI rate limit errors.
7. **Private-company mode**: supplement SEC EDGAR with Crunchbase/PitchBook APIs for private-company financial data.

---

## 13. Conclusion

DeepDiligence demonstrates a practical, auditable multi-agent pattern for investment due diligence. The key architectural contributions are:

- **Strict typed contracts** between agents (Pydantic v2) rather than free-text passing
- **Explicit conflict surfacing** rather than silent resolution
- **Mandatory source traceability** with faithfulness scoring
- **Two-layer design**: specialist evidence collection → portfolio-manager synthesis
- **Thread-safe Streamlit integration** for live pipeline progress without context errors

The system produced memos for 33 companies during development, with composite evaluation scores ranging from 33% (Tesla, data-constrained) to 88% (Apple, well-sourced). The weekly recommendation engine successfully ranked portfolios from STRONG BUY to STRONG SELL based on relative memo quality and analytical content.
