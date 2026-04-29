"""DeepDiligence — Streamlit web application.

Launch with:
    streamlit run app.py

Architecture note
-----------------
The pipeline runs synchronously on the Streamlit main thread via st.status().
The Orchestrator's ThreadPoolExecutor still runs the 4 specialist agents in
parallel internally — we just don't try to update Streamlit UI from those
worker threads. Instead, callbacks write to a plain list (thread-safe for
appends) and st.status.write() is called from the main thread inside the
callbacks fired by the orchestrator's as_completed() loop, which runs on the
main thread.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import streamlit as st

st.set_page_config(
    page_title="DeepDiligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.evaluation.benchmark import get_benchmark_names, load_benchmark
from src.evaluation.faithfulness import score_faithfulness
from src.evaluation.metrics import compute_metrics, metrics_report_text
from src.models.schemas import AgentSubReport, InvestmentMemo
from src.orchestrator import Orchestrator
from src.scheduler.recommender import load_latest_report, load_all_reports, WeeklyReport
from src.scheduler.batch_runner import load_watchlist, run_batch
from src.scheduler.recommender import RecommendationEngine, save_report
from src.scheduler.screener import build_watchlist, ALL_SECTORS, ALL_STYLES

logging.basicConfig(level=logging.INFO)

# ─── Display constants ────────────────────────────────────────────────────────

AGENT_DISPLAY = {
    "financial_analyst":  ("💰", "Financial Analyst"),
    "team_culture":       ("👥", "Team & Culture"),
    "market_competitive": ("📊", "Market & Competitive"),
    "risk_sentiment":     ("⚠️",  "Risk & Sentiment"),
}

def _conf_color(s: float) -> str:
    return "#2ecc71" if s >= 0.70 else ("#f39c12" if s >= 0.50 else "#e74c3c")

def _conf_label(s: float) -> str:
    return "High" if s >= 0.70 else ("Medium" if s >= 0.50 else "Low")

def _conf_badge(score: float) -> str:
    c = _conf_color(score)
    return (f"<span style='background:{c};color:white;padding:2px 8px;"
            f"border-radius:12px;font-size:.85em;font-weight:bold'>"
            f"{_conf_label(score)} ({score:.0%})</span>")


# ─── Session state ────────────────────────────────────────────────────────────

def _init() -> None:
    for k, v in [("memo", None), ("eval_result", None)]:
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Pipeline runner (synchronous, main thread) ───────────────────────────────

def _run_and_display(company: str, ticker: str | None) -> None:
    """Run the full pipeline with a live st.status() progress display.

    Everything happens on the Streamlit main thread, so st.* calls inside
    callbacks are completely safe.  The 4 specialist agents still run in
    parallel via the Orchestrator's ThreadPoolExecutor — we just don't call
    any Streamlit functions from inside those worker threads.
    """
    # Collect status lines here; the on_agent_complete callback is called
    # by the orchestrator's main-thread loop (as_completed), so it IS on
    # the main thread and can call status.write() safely.
    with st.status("🔄 Running analysis — this takes 3–5 minutes…", expanded=True) as status:

        status.write(f"🚀 Starting 4 specialist agents in parallel for **{company}**…")

        def on_agent_start(name: str) -> None:
            # NOTE: this fires from a ThreadPoolExecutor worker thread, so we
            # must NOT call any st.* functions here.  Logging only.
            logging.getLogger(__name__).info("Agent started: %s", name)

        def on_agent_complete(name: str, report: AgentSubReport) -> None:
            em, lbl = AGENT_DISPLAY.get(name, ("🤖", name))
            conf = report.confidence_score
            color = _conf_color(conf)
            if report.findings:
                status.write(
                    f"✅ {em} **{lbl}** — {len(report.findings)} findings · "
                    f"<span style='color:{color};font-weight:bold'>{conf:.0%} confidence</span>",
                    unsafe_allow_html=True,
                )
            else:
                status.write(f"⚠️ {em} **{lbl}** — no findings (data limited)")

        def on_synthesis_start() -> None:
            status.write("🧠 **Synthesis Agent** — reconciling all findings…")

        try:
            orch = Orchestrator(
                company_name=company,
                ticker=ticker or None,
                on_agent_start=on_agent_start,
                on_agent_complete=on_agent_complete,
                on_synthesis_start=on_synthesis_start,
            )
            memo = orch.run()
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
            st.session_state.memo = memo

        except Exception as exc:
            status.update(label=f"❌ Pipeline failed: {exc}", state="error")
            st.session_state.memo = None
            return

    st.rerun()   # re-render page to show the memo


# ─── Memo renderer ────────────────────────────────────────────────────────────

def _render_section(section: Any, source_registry: dict | None = None) -> None:
    st.markdown(
        f"### {section.title} &nbsp; {_conf_badge(section.confidence_score)}",
        unsafe_allow_html=True,
    )
    st.markdown(section.content)

    if section.conflicting_claims:
        with st.expander(f"⚠️ {len(section.conflicting_claims)} cross-agent conflict(s)"):
            for c in section.conflicting_claims:
                st.warning(f"**{c.description}**")
                col_a, col_b = st.columns(2)
                col_a.markdown(f"**Claim A:** {c.claim_a.text}")
                col_b.markdown(f"**Claim B:** {c.claim_b.text}")

    if section.claims:
        with st.expander(f"📋 {len(section.claims)} supporting claims"):
            for claim in section.claims:
                n = len(claim.source_ids)
                if n == 0:
                    source_links = "⚠️ no source"
                elif source_registry:
                    links = []
                    for i, sid in enumerate(claim.source_ids, 1):
                        src = source_registry.get(sid)
                        if src and src.get("url"):
                            title = src.get("title") or src["url"]
                            links.append(f"[{i}]({src['url']} \"{title}\")")
                        else:
                            links.append(f"[{i}]")
                    source_links = " ".join(links)
                else:
                    source_links = f"({n} source{'s' if n != 1 else ''})"
                st.markdown(
                    f"- [{claim.confidence:.0%}] {claim.text} {source_links}",
                    unsafe_allow_html=False,
                )


def _render_memo(memo: InvestmentMemo) -> None:
    st.success(
        f"✅ **{memo.company_name}** — overall confidence **{memo.overall_confidence:.0%}** "
        f"· {len(memo.sections)} sections · {memo.metadata.get('total_findings', '?')} findings"
    )

    tab_labels = (
        ["📋 Executive Summary"]
        + [s.title for s in memo.sections]
        + ["📊 Stats & Sources", "⬇️ Export"]
    )
    tabs = st.tabs(tab_labels)

    # Executive summary
    with tabs[0]:
        st.markdown("## Executive Summary")
        st.markdown(memo.executive_summary)
        hi = memo.metadata.get("investment_highlights", [])
        ri = memo.metadata.get("investment_risks", [])
        if hi or ri:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ✅ Investment Highlights")
                for h in hi:
                    st.markdown(f"- {h}")
            with c2:
                st.markdown("### ⚠️ Key Risks")
                for r in ri:
                    st.markdown(f"- {r}")

    # Per-section tabs
    source_registry = memo.metadata.get("source_registry", {})
    for i, section in enumerate(memo.sections):
        with tabs[i + 1]:
            _render_section(section, source_registry)

    # Stats tab
    with tabs[-2]:
        meta = memo.metadata
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Findings", meta.get("total_findings", "—"))
        c2.metric("Sources",  meta.get("total_sources",  "—"))
        c3.metric("Agents",   meta.get("agent_count",    "—"))
        elapsed = meta.get("elapsed_seconds")
        c4.metric("Run time", f"{elapsed}s" if elapsed else "—")

        st.markdown("### Agent confidence")
        for aname, conf in meta.get("specialist_confidences", {}).items():
            em, lbl = AGENT_DISPLAY.get(aname, ("🤖", aname))
            color = _conf_color(conf)
            st.markdown(
                f"**{em} {lbl}** &nbsp;"
                f"<span style='color:{color};font-weight:bold'>{conf:.0%}</span>",
                unsafe_allow_html=True,
            )
            st.progress(int(conf * 100))

        st.markdown("### Source traceability (faithfulness)")
        faith = score_faithfulness(memo)
        fc = _conf_color(faith.overall_faithfulness)
        st.markdown(
            f"**Faithfulness:** <span style='color:{fc};font-weight:bold'>"
            f"{faith.overall_faithfulness:.0%} [{faith.grade}]</span> "
            f"— {faith.sourced_claims}/{faith.total_claims} claims have at least one source",
            unsafe_allow_html=True,
        )

    # Export tab
    with tabs[-1]:
        memo_json = memo.model_dump_json(indent=2)
        safe_name = memo.company_name.lower().replace(" ", "_")
        st.download_button(
            "⬇️ Download memo JSON",
            data=memo_json,
            file_name=f"memo_{safe_name}.json",
            mime="application/json",
        )
        st.markdown("**Raw preview (first 3 000 chars)**")
        st.code(memo_json[:3000] + ("\n…" if len(memo_json) > 3000 else ""), language="json")


def _render_eval_sidebar(memo: InvestmentMemo) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Evaluate vs benchmark")

    # Find the benchmark that matches the company that was actually analysed.
    # Only show the evaluation option when a matching benchmark exists — comparing
    # an Apple memo against a Microsoft benchmark would produce meaningless scores.
    available = get_benchmark_names()
    matched = next(
        (n for n in available if memo.company_name.lower() in n.lower()
         or n.lower() in memo.company_name.lower()),
        None,
    )

    if not matched:
        st.sidebar.info(
            f"No benchmark profile found for **{memo.company_name}**. "
            f"Benchmarks exist for: {', '.join(available)}."
        )
        return

    st.sidebar.markdown(f"Comparing memo against: **{matched}** benchmark")

    if st.sidebar.button("Run evaluation"):
        try:
            bm = load_benchmark(matched)
            st.session_state.eval_result = compute_metrics(memo, bm)
        except FileNotFoundError as e:
            st.sidebar.error(str(e))

    ev = st.session_state.get("eval_result")
    if ev:
        st.sidebar.markdown(f"**Score:** {ev.composite_score:.0%} [{ev.grade}]")
        st.sidebar.markdown(
            f"Facts: {ev.fact_coverage:.0%} · Risks: {ev.risk_coverage:.0%}"
        )
        if ev.faithfulness:
            st.sidebar.markdown(f"Faithful: {ev.faithfulness.overall_faithfulness:.0%}")
        with st.sidebar.expander("Full evaluation report"):
            st.text(metrics_report_text(ev))


# ─── Weekly report renderer ───────────────────────────────────────────────────

RATING_COLOR = {
    "STRONG BUY":  "#27ae60",
    "BUY":         "#2ecc71",
    "HOLD":        "#f39c12",
    "SELL":        "#e74c3c",
    "STRONG SELL": "#c0392b",
}
RATING_EMOJI = {
    "STRONG BUY":  "🟢🟢",
    "BUY":         "🟢",
    "HOLD":        "🟡",
    "SELL":        "🔴",
    "STRONG SELL": "🔴🔴",
}


def _render_weekly_report(report: WeeklyReport) -> None:
    st.markdown(f"## 📈 Weekly Report — {report.week_of}")
    st.caption(
        f"Generated {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')} · "
        f"{report.universe_size} companies analysed · model: {report.model_used}"
    )

    # Top picks + avoid banner
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**✅ Top picks:** {', '.join(report.top_picks)}")
    with col2:
        if report.avoid:
            st.error(f"**❌ Avoid:** {', '.join(report.avoid)}")

    # Macro commentary
    st.info(f"**📊 Macro view:** {report.macro_commentary}")

    # Sector views
    if report.sector_views:
        st.markdown("### 🏭 Sector Views")
        cols = st.columns(min(len(report.sector_views), 4))
        for i, (sector, view) in enumerate(report.sector_views.items()):
            arrow = "↑" if "Bullish" in view else ("↓" if "Bearish" in view else "→")
            cols[i % 4].metric(sector, f"{arrow} {view}")

    st.markdown("### 📋 Rankings")

    # Summary table
    for r in report.ratings:
        color  = RATING_COLOR.get(r.rating, "#888")
        emoji  = RATING_EMOJI.get(r.rating, "")
        weight = f"{r.suggested_weight_pct:.1f}%" if r.suggested_weight_pct > 0 else "—"

        with st.expander(
            f"#{r.rank}  **{r.ticker}** — {r.company_name}   "
            f"{emoji} {r.rating}   weight: {weight}   conf: {r.confidence:.0%}"
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"<span style='background:{color};color:white;padding:4px 12px;"
                    f"border-radius:8px;font-weight:bold;font-size:1.1em'>"
                    f"{emoji} {r.rating}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Sector:** {r.sector}")
                st.markdown(f"**Suggested weight:** {weight}")
                st.markdown(f"**Confidence:** {r.confidence:.0%}")
            with c2:
                st.markdown(f"🐂 **Bull case:** {r.bull_case}")
                st.markdown(f"🐻 **Bear case:** {r.bear_case}")
            st.markdown(f"**Rationale:** {r.rationale}")

    # Historical reports
    all_reports = load_all_reports()
    if len(all_reports) > 1:
        st.markdown("---")
        st.markdown("### 🕰 Historical Reports")
        for old in all_reports[1:]:
            with st.expander(f"Week {old.week_of} — Top picks: {', '.join(old.top_picks)}"):
                for r in old.ratings:
                    em = RATING_EMOJI.get(r.rating, "")
                    st.markdown(f"- #{r.rank} **{r.ticker}** {em} {r.rating}")


# ─── Weekly pipeline runner ───────────────────────────────────────────────────

def _run_weekly_and_display(
    styles: list[str],
    sectors: list[str],
    top_n: int,
    use_screener: bool,
    screener_criteria: str,
) -> None:
    """Build watchlist from preferences, run batch pipeline, generate report."""

    with st.status("📈 Generating weekly report…", expanded=True) as status:

        # Step 1 — build watchlist
        status.write("🔍 Building stock watchlist from your preferences…")
        selected = build_watchlist(
            use_screener=use_screener,
            screener_criteria=screener_criteria,
            styles=styles,
            sectors=sectors,
            top_n=top_n,
        )

        if not selected:
            status.update(label="❌ No stocks matched your filters", state="error")
            return

        style_label  = ", ".join(styles)  if styles  else "All styles"
        sector_label = ", ".join(sectors) if sectors else "All sectors"
        mode_label   = f"S&P 500 screener ({screener_criteria})" if use_screener else "Curated universe"

        status.write(
            f"✅ Selected **{len(selected)} stocks** · {style_label} · "
            f"{sector_label} · {mode_label}"
        )
        for s in selected:
            status.write(f"  — **{s['ticker']}** {s['company']} ({s['sector']})")

        # Step 2 — run pipeline on each stock
        status.write(f"\n🚀 Running due diligence pipeline on {len(selected)} companies…")
        memos: dict = {}
        errors: list[str] = []

        for stock in selected:
            try:
                status.write(f"⏳ Analysing **{stock['ticker']}** — {stock['company']}…")
                from src.scheduler.batch_runner import _run_one
                stk, memo = _run_one(stock, force_refresh=False)
                if memo:
                    memos[stock["ticker"]] = memo
                    conf = memo.overall_confidence
                    color = _conf_color(conf)
                    status.write(
                        f"✅ **{stock['ticker']}** — {len(memo.sections)} sections · "
                        f"<span style='color:{color}'>{conf:.0%} confidence</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    errors.append(stock["ticker"])
                    status.write(f"⚠️ **{stock['ticker']}** — pipeline failed, skipping")
            except Exception as exc:
                errors.append(stock["ticker"])
                status.write(f"⚠️ **{stock['ticker']}** — error: {str(exc)[:80]}")

        if not memos:
            status.update(label="❌ All pipelines failed — no memos to rank", state="error")
            return

        # Step 3 — generate recommendations
        status.write(f"\n🧠 Ranking {len(memos)} companies via portfolio manager LLM…")
        try:
            engine = RecommendationEngine()
            report = engine.generate(memos)
            path   = save_report(report)
            status.update(
                label=f"✅ Weekly report complete — {len(report.ratings)} stocks ranked!",
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            status.update(label=f"❌ Recommendation engine failed: {exc}", state="error")
            return

    st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _init()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔍 DeepDiligence")
        st.markdown("*Multi-Agent Investment Due Diligence*")
        st.markdown("---")

        # Page selector
        page = st.radio(
            "View",
            ["🔍 Single Company", "📈 Weekly Rankings"],
            label_visibility="collapsed",
        )
        st.markdown("---")

        if page == "🔍 Single Company":
            company_input = st.text_input("Company name", value="Apple Inc")
            raw_ticker = st.text_input("Ticker (optional)", value="AAPL").strip().upper()
            ticker_input = raw_ticker or None
            run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

            if st.session_state.memo:
                _render_eval_sidebar(st.session_state.memo)
                if st.sidebar.button("🔄 New analysis"):
                    st.session_state.memo = None
                    st.session_state.eval_result = None
                    st.rerun()
        else:
            run_btn = False
            st.markdown("### 🎯 Investment Preferences")

            invest_style = st.multiselect(
                "Investment style",
                ALL_STYLES,
                default=[],
                placeholder="All styles",
                help="Leave empty to include all styles",
            )

            sectors = st.multiselect(
                "Sectors",
                ALL_SECTORS,
                default=[],
                placeholder="All sectors",
                help="Leave empty to include all sectors",
            )

            top_n = st.slider("Number of stocks to analyse", min_value=3, max_value=20, value=10)

            st.markdown("---")
            st.markdown("### 🔍 Stock Selection")
            use_screener = st.toggle(
                "Use live S&P 500 screener",
                value=False,
                help="Pull this week's top movers from S&P 500 using real price/volume data",
            )

            if use_screener:
                screener_criteria = st.selectbox(
                    "Rank stocks by",
                    ["Price Change", "Volume", "Price Change + Volume"],
                    help="Price Change = biggest movers · Volume = most traded",
                )
            else:
                screener_criteria = "Price Change"
                st.caption("Using curated stock universe filtered by your preferences above")

            st.markdown("---")
            weekly_btn = st.button(
                "📈 Generate Weekly Report",
                use_container_width=True,
                type="primary",
            )

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "<small>OpenAI GPT-5.4-mini / GPT-5.5 · DeepDiligence v1 · Cornell ENMGT5400</small>",
            unsafe_allow_html=True,
        )

    # ── Main area ────────────────────────────────────────────────────────────
    if page == "📈 Weekly Rankings":
        if weekly_btn:
            _run_weekly_and_display(
                styles=invest_style,
                sectors=sectors,
                top_n=top_n,
                use_screener=use_screener,
                screener_criteria=screener_criteria,
            )
            return

        report = load_latest_report()
        if report:
            _render_weekly_report(report)
        else:
            st.markdown("# 📈 Weekly Rankings")
            st.info(
                "Configure your preferences in the sidebar and click "
                "**Generate Weekly Report** to get started."
            )
            st.markdown("**Available universe:** 60+ stocks across all sectors and styles.")
        return

    # Single company page
    if run_btn:
        st.session_state.memo = None
        st.session_state.eval_result = None
        _run_and_display(company_input.strip(), ticker_input)
        return

    memo = st.session_state.get("memo")
    if memo:
        _render_memo(memo)
    else:
        # Landing page
        st.markdown("# 🔍 DeepDiligence")
        st.markdown("**Automated investment due diligence powered by four specialized AI agents.**")
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("### 💰 Financial\nSEC filings · revenue · financial health")
        c2.markdown("### 👥 Team & Culture\nLeadership · hiring · culture signals")
        c3.markdown("### 📊 Market\nCompetitors · positioning · growth")
        c4.markdown("### ⚠️ Risk\nLitigation · regulatory · reputational")
        st.markdown("---")
        st.info("👈 Enter a company name and ticker in the sidebar, then click **Run Analysis**.")

        st.markdown("### 📚 Quick-start")
        cols = st.columns(5)
        quick = [("Apple Inc","AAPL"),("Microsoft","MSFT"),("NVIDIA","NVDA"),
                 ("Tesla","TSLA"),("Amazon","AMZN")]
        for i, (bname, bticker) in enumerate(quick):
            with cols[i]:
                if st.button(bname, key=f"qs_{bticker}"):
                    st.session_state.memo = None
                    st.session_state.eval_result = None
                    _run_and_display(bname, bticker)


if __name__ == "__main__":
    main()
