"""Patch existing memo caches to add source_registry to metadata.

Reads the 4 agent sub-report caches for each memo, builds the source registry
(source_id → {url, title, snippet, source_type}), and writes it into the
memo's metadata without changing anything else.

Run once:
    python3.11 scripts/patch_source_registry.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import CACHE_DIR

MEMOS_DIR    = CACHE_DIR / "memos"
AGENTS_DIR   = CACHE_DIR / "agents"
AGENT_NAMES  = ["financial_analyst", "team_culture", "market_competitive", "risk_sentiment"]


def _cache_key(agent_name: str, company_name: str, ticker: str) -> str:
    raw = f"{agent_name}|{company_name}|{ticker}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_agent_report(agent_name: str, company_name: str, ticker: str) -> dict | None:
    key  = _cache_key(agent_name, company_name, ticker)
    path = AGENTS_DIR / agent_name / f"{key}_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def build_registry_from_agents(company_name: str, ticker: str) -> dict:
    """Load all 4 agent reports and merge their sources into one registry."""
    registry: dict[str, dict] = {}
    for agent_name in AGENT_NAMES:
        report = load_agent_report(agent_name, company_name, ticker)
        if not report:
            continue
        for src in report.get("sources", []):
            sid = src.get("id", "")
            if not sid or not src.get("url"):
                continue
            entry = {
                "url":         src["url"],
                "title":       src.get("title", src["url"])[:200],
                "snippet":     src.get("snippet", "")[:200],
                "source_type": src.get("source_type", "other"),
            }
            namespaced = f"{agent_name}::{sid}"
            registry[namespaced] = entry
            registry[sid] = entry          # bare ID for LLM-generated references
    return registry


def patch_all() -> None:
    memo_paths = sorted(MEMOS_DIR.glob("*_memo.json"))
    if not memo_paths:
        print("No memo files found.")
        return

    patched = 0
    skipped = 0
    no_sources = 0

    for memo_path in memo_paths:
        with open(memo_path) as f:
            memo = json.load(f)

        # Already patched
        if memo.get("metadata", {}).get("source_registry"):
            skipped += 1
            continue

        ticker       = memo_path.stem.replace("_memo", "")
        company_name = memo.get("company_name", ticker)

        registry = build_registry_from_agents(company_name, ticker)

        if not registry:
            print(f"  ⚠  {ticker:6}  {company_name:30} — no agent caches found, skipping")
            no_sources += 1
            continue

        memo.setdefault("metadata", {})["source_registry"] = registry

        with open(memo_path, "w") as f:
            json.dump(memo, f, indent=2)

        patched += 1
        print(f"  ✅ {ticker:6}  {company_name:30} — {len(registry)} source entries added")

    print(f"\nDone. Patched: {patched}  |  Already OK: {skipped}  |  No agent cache: {no_sources}")


if __name__ == "__main__":
    patch_all()
