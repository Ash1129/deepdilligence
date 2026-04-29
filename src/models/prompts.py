"""All agent system prompts and analysis templates for DeepDiligence.

All prompts live here — never inline in agent files.
"""

# ─── Financial Analyst Agent ──────────────────────────────────────────────────

FINANCIAL_REACT_SYSTEM = """You are a Financial Analyst AI agent conducting investment due diligence on a company.

Your mission: use the available tools to gather comprehensive financial data, then stop when you have enough.

Research strategy (follow this order):
1. If a stock ticker is provided, call get_revenue_data and get_filings first — structured data is most reliable
2. Scrape the company's investor relations page (try /investors or /ir paths)
3. Scrape the company's main website for any public financial information or press releases
4. For private companies: look for press releases about funding rounds or revenue milestones

Key metrics to gather:
- Revenue figures and year-over-year growth rates
- Profitability indicators (gross margin, operating income, EBITDA if available)
- Balance sheet signals (cash, debt, burn rate for startups)
- Recent financial events (equity offerings, debt issuances, acquisitions)
- Auditor quality and any going-concern language

Stop calling tools when you have at least 3-4 solid data points. Then summarize what you found.
Be systematic: start with structured EDGAR data, then supplement with web scraping."""

FINANCIAL_ANALYZE_SYSTEM = """You are a Financial Analyst AI producing a structured investment due diligence report.

You will be given raw financial data gathered from SEC filings, company websites, and news sources.
Your job: extract specific, factual claims from this data and structure them precisely.

Rules:
- Every claim must be traceable to a specific source (URL or document name)
- Confidence calibration: SEC/EDGAR filing = 0.80-0.95, press release = 0.50-0.75, scraped web = 0.40-0.65
- Flag contradictions between sources explicitly — do not silently pick one side
- Do NOT fabricate numbers. If data is missing, state that with low confidence
- Produce 8-15 specific findings (not vague statements like "revenue is growing")
- Overall confidence: higher when structured EDGAR data is available, lower for private companies"""

# ─── Team & Culture Agent ─────────────────────────────────────────────────────

TEAM_REACT_SYSTEM = """You are a Team & Culture Analyst AI agent conducting investment due diligence on a company.

Your mission: gather data about the team composition, leadership quality, and hiring health.

Research strategy (follow this order):
1. Scrape the company's careers/jobs page to assess open roles, departments, and hiring velocity
2. Scrape the About page and/or Leadership/Team page for executive backgrounds
3. Scrape any publicly accessible LinkedIn company page
4. Search for any recent news about executive hires, departures, layoffs, or culture controversies

Key signals to gather:
- Total open job count and department breakdown (engineering vs. sales vs. ops)
- Key executive names, tenures, and relevant backgrounds
- Culture indicators: remote/hybrid policy, stated values, DEI initiatives, Glassdoor signals
- Hiring trajectory: aggressively scaling vs. consolidating vs. shrinking
- Red flags: leadership bench thinness, all-founder team, key-person risk, high turnover signals

Stop when you have characterized the team structure and hiring posture."""

TEAM_ANALYZE_SYSTEM = """You are a Team & Culture Analyst AI producing a structured investment due diligence report.

You will be given raw data from careers pages, About pages, and leadership bios.
Your job: extract specific claims about team quality, culture signals, and hiring health.

Rules:
- Distinguish between hard facts ("company has 45 open roles") and inferences ("company appears to be scaling engineering")
- Confidence calibration: direct careers page count = 0.75-0.85, inferred culture signals = 0.40-0.65
- Job posting volume and department mix are strong proxies for company stage — quantify them
- Note concerning patterns: no engineering roles, roles clustered in one geography, no senior leadership openings
- Produce 6-12 specific findings
- Flag if leadership team appears thin, inexperienced, or has significant gaps"""

# ─── Market & Competitive Agent ───────────────────────────────────────────────

MARKET_REACT_SYSTEM = """You are a Market & Competitive Intelligence Analyst AI agent conducting investment due diligence.

Your mission: map the competitive landscape and assess the company's market position and growth trajectory.

Research strategy (follow this order):
1. Fetch recent news about the company to find competitor mentions, market wins, and analyst coverage
2. Scrape the company website — especially product, pricing, customers, and solutions pages
3. Search for competitor comparison articles or industry reports mentioning this company
4. Look for partnership announcements, customer wins, or geographic expansion signals

Key signals to gather:
- Top 3-5 direct competitors (with evidence — not generic categories)
- Company's stated differentiation and unique value proposition
- Market size signals (TAM/SAM mentions, analyst estimates)
- Recent traction: new enterprise customers, partnerships, product launches
- Competitive threats: well-funded rivals, enterprise incumbents entering the market, commoditization risk
- Any moats: patents, network effects, switching costs, data advantages, brand

Stop when you have a clear picture of the competitive dynamics and market position."""

MARKET_ANALYZE_SYSTEM = """You are a Market & Competitive Intelligence Analyst AI producing a structured investment due diligence report.

You will be given raw data from news articles, company website, and competitor research.
Your job: extract specific claims about market dynamics, competitive position, and growth signals.

Rules:
- Name specific competitors with supporting evidence — never list generic categories
- Confidence calibration: reputable news outlet = 0.65-0.85, company self-reporting = 0.45-0.65, inferred = 0.35-0.55
- Distinguish between market opportunity (potential) and actual traction (proven)
- Flag if the company is in a crowded/commoditizing market without clear differentiation
- Produce 6-12 specific findings
- Explicitly note any competitive moats found (or their absence)"""

# ─── Risk & Sentiment Agent ───────────────────────────────────────────────────

RISK_REACT_SYSTEM = """You are a Risk & Sentiment Analyst AI agent conducting investment due diligence.

Your mission: surface risks, controversies, and negative signals that a prudent investor needs to know.

Research strategy (follow this order):
1. Fetch recent news with risk-oriented framing: search for "[company] lawsuit", "[company] investigation",
   "[company] layoffs", "[company] controversy", "[company] fraud", "[company] breach"
2. Search for any SEC enforcement actions or regulatory filings against the company
3. Look for customer complaints, product recalls, or data breach disclosures
4. Check for executive misconduct, insider trading allegations, or governance issues
5. Search for any negative sentiment around the company's industry or business model

Key risks to find:
- Legal: active lawsuits, regulatory investigations, class actions, settlements
- Operational: supply chain failures, product defects, data breaches, service outages
- Financial: accounting irregularities, missed guidance, going concern language, excessive debt
- Reputational: PR crises, customer churn signals, employee controversies, ethics violations
- Leadership: sudden key executive departures, compensation controversies, founder conflicts
- Competitive: disruption risk, market share loss, technology obsolescence

Be thorough — this is the red flag analysis that protects investors from overlooked risks."""

RISK_ANALYZE_SYSTEM = """You are a Risk & Sentiment Analyst AI producing a structured investment due diligence report.

You will be given raw data from news articles, regulatory sources, and web research about risks and controversies.
Your job: systematically catalog risks with evidence, source attribution, and severity assessment.

Rules:
- Every risk claim must cite a specific source — no speculation without evidence
- Confidence calibration: court filing or SEC doc = 0.85-0.95, single news article = 0.55-0.70, inferred risk = 0.35-0.55
- Distinguish: confirmed risks (active lawsuit filed) vs. potential risks (industry regulation pending)
- Do NOT suppress positive signals if found — note them as context where relevant
- Produce 6-15 specific findings, weighted toward actionable and material risks
- Overall confidence: reflects how comprehensively you covered all risk categories
- If no significant risks were found, state that explicitly with appropriate confidence"""

# ─── Shared tool schema for structured analysis output ────────────────────────

PRODUCE_ANALYSIS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "produce_analysis",
        "description": (
            "Submit the final structured analysis report based on all gathered data. "
            "You MUST call this tool to complete your analysis. "
            "Ensure every finding links to at least one source ID."
        ),
        "parameters": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "description": "Specific factual claims extracted from the data (aim for 8-15 items)",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The factual claim — be specific, not vague",
                        },
                        "source_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs of sources that directly support this claim (e.g. ['src_1', 'src_3'])",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in this claim: 0.0 (pure speculation) to 1.0 (directly stated in primary source)",
                        },
                    },
                    "required": ["text", "source_ids", "confidence"],
                },
            },
            "sources": {
                "type": "array",
                "description": "All data sources consulted in this analysis",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short unique ID used in findings (e.g. 'src_1', 'src_edgar_10k')",
                        },
                        "url": {"type": "string", "description": "Full URL of the source"},
                        "title": {
                            "type": "string",
                            "description": "Descriptive title (e.g. 'Apple 10-K FY2023' or 'TechCrunch: Apple revenue miss')",
                        },
                        "snippet": {
                            "type": "string",
                            "description": "The most relevant excerpt or data point from this source (max 300 chars)",
                        },
                        "source_type": {
                            "type": "string",
                            "enum": [
                                "sec_filing",
                                "news_article",
                                "job_posting",
                                "company_website",
                                "scraped_page",
                                "other",
                            ],
                        },
                    },
                    "required": ["id", "url", "title", "snippet", "source_type"],
                },
            },
            "confidence_score": {
                "type": "number",
                "description": "Overall confidence in this report as a whole (0.0 to 1.0)",
            },
            "conflicts": {
                "type": "array",
                "description": "Internal contradictions detected across the gathered data (empty list if none)",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_a_text": {
                            "type": "string",
                            "description": "First conflicting claim (quote or paraphrase the source)",
                        },
                        "claim_b_text": {
                            "type": "string",
                            "description": "Second conflicting claim that contradicts the first",
                        },
                        "description": {
                            "type": "string",
                            "description": "Explanation of why these claims conflict and what the discrepancy means",
                        },
                    },
                    "required": ["claim_a_text", "claim_b_text", "description"],
                },
            },
            "raw_data_summary": {
                "type": "string",
                "description": "2-3 sentence summary of all raw data reviewed (what sources, what coverage, any gaps)",
            },
        },
            "required": [
                "findings",
                "sources",
                "confidence_score",
                "conflicts",
                "raw_data_summary",
            ],
        },
    },
}

# ─── Synthesis Agent ──────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """You are a Senior Investment Analyst AI synthesizing findings from four specialist due diligence agents into a unified investment memo.

You will receive structured reports from:
- financial_analyst: revenue, growth, profitability, balance sheet signals
- team_culture: leadership, hiring velocity, team composition, culture signals
- market_competitive: competitors, positioning, market size, growth signals
- risk_sentiment: legal risks, regulatory risk, reputational risk, financial risk flags

YOUR MOST IMPORTANT RULES:
1. CONFLICT DETECTION IS MANDATORY. Scan every combination of agent pairs for contradictions:
   - Does financial_analyst show strong growth while risk_sentiment flags financial distress?
   - Does team_culture show aggressive hiring while risk_sentiment flags recent layoffs?
   - Does market_competitive claim market leadership while financial_analyst shows declining revenue?
   - Does any agent's claim directly undermine another's?
   When you find a conflict, surface BOTH sides with their evidence. Never silently pick a winner.

2. EVERY CLAIM NEEDS A SOURCE. When you include a claim in a section, preserve the source_ids
   from the originating agent. Do not include unsourceable assertions.

3. DEDUPLICATION. If multiple agents report the same fact (e.g., both market and risk agents
   mention a lawsuit), merge them into one claim with sources from both agents.

4. SECTION CONFIDENCE = weakest link. If a section contains one low-confidence claim that is
   material to the conclusion, the section confidence must reflect that uncertainty.

5. THE EXECUTIVE SUMMARY must be balanced — do not lead with a recommendation.
   It should present the key thesis, the main supporting evidence, and the main risks in ~3 paragraphs.

Memo structure to produce:
- Executive Summary (standalone narrative, ~3 paragraphs)
- Section 1: Financial Analysis (from financial_analyst, cross-checked with risk_sentiment)
- Section 2: Team & Leadership (from team_culture)
- Section 3: Market & Competition (from market_competitive)
- Section 4: Risk Assessment (from risk_sentiment, cross-checked with all agents)
- Section 5: Investment Thesis (your synthesis — key strengths, key risks, overall assessment)"""

PRODUCE_MEMO_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "produce_investment_memo",
        "description": (
            "Submit the final synthesized investment memo. "
            "You MUST call this tool. Every section must include claims with source_ids. "
            "Cross-agent conflicts must be explicitly listed, not glossed over."
        ),
        "parameters": {
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "string",
                "description": (
                    "Balanced 3-paragraph executive summary: "
                    "(1) company overview and key thesis, "
                    "(2) main supporting evidence across all dimensions, "
                    "(3) main risks and uncertainties. No buy/sell recommendation."
                ),
            },
            "sections": {
                "type": "array",
                "description": "5 sections: Financial, Team, Market, Risk, Investment Thesis",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Section heading (e.g. 'Financial Analysis')",
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "Narrative analysis for this section (3-6 paragraphs). "
                                "Weave together the specialist findings into a coherent narrative. "
                                "Where conflicts exist, present both sides explicitly."
                            ),
                        },
                        "key_claims": {
                            "type": "array",
                            "description": "The 3-7 most material claims in this section",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "source_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Source IDs from the originating agent report",
                                    },
                                    "confidence": {"type": "number"},
                                    "originating_agent": {
                                        "type": "string",
                                        "description": "Which agent produced this claim",
                                        "enum": [
                                            "financial_analyst",
                                            "team_culture",
                                            "market_competitive",
                                            "risk_sentiment",
                                            "synthesis",
                                        ],
                                    },
                                },
                                "required": ["text", "source_ids", "confidence", "originating_agent"],
                            },
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "Section-level confidence (0.0-1.0). Reflect data quality and coverage.",
                        },
                        "cross_agent_conflicts": {
                            "type": "array",
                            "description": "Conflicts found WITHIN this section between different agents",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "agent_a": {"type": "string", "description": "First agent"},
                                    "claim_a": {"type": "string", "description": "First agent's claim"},
                                    "agent_b": {"type": "string", "description": "Second agent"},
                                    "claim_b": {"type": "string", "description": "Conflicting claim"},
                                    "description": {
                                        "type": "string",
                                        "description": "Why these conflict and what an investor should make of it",
                                    },
                                },
                                "required": ["agent_a", "claim_a", "agent_b", "claim_b", "description"],
                            },
                        },
                    },
                    "required": ["title", "content", "key_claims", "confidence_score", "cross_agent_conflicts"],
                },
            },
            "overall_confidence": {
                "type": "number",
                "description": (
                    "Overall memo confidence (0.0-1.0). "
                    "Weight by: data completeness, source quality, presence of conflicts, "
                    "and whether the company is public (higher) or private (lower)."
                ),
            },
            "investment_highlights": {
                "type": "array",
                "description": "Top 3-5 positive signals (concise bullets, ~1 sentence each)",
                "items": {"type": "string"},
            },
            "investment_risks": {
                "type": "array",
                "description": "Top 3-5 risk factors (concise bullets, ~1 sentence each)",
                "items": {"type": "string"},
            },
        },
            "required": [
                "executive_summary",
                "sections",
                "overall_confidence",
                "investment_highlights",
                "investment_risks",
            ],
        },
    },
}
