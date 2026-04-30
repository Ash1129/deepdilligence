export const API_BASE = "http://localhost:8000";

// ── Shared types ─────────────────────────────────────────────────────────
export interface Source {
  id: string;
  url: string;
  source_type: "sec_edgar" | "news" | "web" | "careers" | "other";
  snippet: string;
  retrieved_at: string;
}

export interface Claim {
  text: string;
  confidence: number;
  source_ids: string[];
  reasoning?: string;
}

export interface ConflictingClaim {
  description: string;
  claim_a: Claim;
  claim_b: Claim;
  resolution?: string;
}

export interface SynthesizedSection {
  title: string;
  content: string;
  claims: Claim[];
  conflicting_claims: ConflictingClaim[];
  confidence_score: number;
}

export interface InvestmentMemo {
  company_name: string;
  executive_summary: string;
  sections: SynthesizedSection[];
  overall_confidence: number;
  metadata: {
    total_findings: number;
    total_sources: number;
    agent_count: number;
    elapsed_seconds: number;
    specialist_confidences: Record<string, number>;
    investment_highlights: string[];
    investment_risks: string[];
    source_registry?: Record<
      string,
      {
        id?: string;
        original_id?: string;
        originating_agent?: string;
        url: string;
        title: string;
        snippet: string;
        source_type: string;
        retrieved_at?: string;
      }
    >;
    verification?: {
      company_name: string;
      total_claims: number;
      supported_claims: number;
      weak_claims: number;
      unsupported_claims: number;
      missing_source_claims: number;
      unresolved_source_claims: number;
      overall_score: number;
      hallucination_risk: number;
      grade: string;
      per_claim: Array<{
        section_title: string;
        claim_text: string;
        source_ids: string[];
        status:
          | "supported"
          | "weak"
          | "unsupported"
          | "missing_source"
          | "unresolved_source";
        support_score: number;
        matched_terms: string[];
        missing_numbers: string[];
        reason: string;
      }>;
    };
  };
}

export type Rating = "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL";

export interface StockRating {
  company_name: string;
  ticker: string;
  sector: string;
  rating: Rating;
  rank: number;
  bull_case: string;
  bear_case: string;
  rationale: string;
  suggested_weight_pct: number;
  confidence: number;
}

export interface WeeklyReport {
  generated_at: string;
  week_of: string;
  universe_size: number;
  ratings: StockRating[];
  top_picks: string[];
  avoid: string[];
  macro_commentary: string;
  sector_views: Record<string, string>;
  model_used: string;
}

export interface Stock {
  ticker: string;
  company: string;
  sector: string;
  styles: string[];
}

export interface UniverseResponse {
  stocks: Stock[];
  all_styles: string[];
  all_sectors: string[];
}

export interface WeeklyReportRequest {
  styles: string[];
  sectors: string[];
  top_n: number;
  use_screener: boolean;
  screener_criteria: "Price Change" | "Volume" | "Price Change + Volume";
}

// ── REST helpers ─────────────────────────────────────────────────────────
export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

// ── SSE streaming ────────────────────────────────────────────────────────
export async function streamSSE(
  path: string,
  body: unknown,
  onEvent: (event: any) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`Stream failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (!payload) continue;
      try {
        onEvent(JSON.parse(payload));
      } catch {
        // skip malformed
      }
    }
  }
}

// ── Display helpers ──────────────────────────────────────────────────────
export const AGENT_META: Record<string, { label: string; icon: string }> = {
  financial_analyst: { label: "Financial Analyst", icon: "💰" },
  team_culture: { label: "Team & Culture", icon: "👥" },
  market_competitive: { label: "Market & Competitive", icon: "📊" },
  risk_sentiment: { label: "Risk & Sentiment", icon: "⚠️" },
};

export function confTier(c: number): "high" | "mid" | "low" {
  if (c >= 0.7) return "high";
  if (c >= 0.5) return "mid";
  return "low";
}
