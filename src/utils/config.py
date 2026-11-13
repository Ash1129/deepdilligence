"""Configuration management for DeepDiligence."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable (1/true/yes/on → True)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_news_api_key() -> str:
    """Return the NewsAPI key from environment."""
    key = os.getenv("NEWS_API_KEY", "")
    if not key:
        raise ValueError("NEWS_API_KEY not set in environment")
    return key


def get_sec_edgar_user_agent() -> str:
    """Return the SEC EDGAR User-Agent string from environment."""
    ua = os.getenv("SEC_EDGAR_USER_AGENT", "")
    if not ua:
        raise ValueError("SEC_EDGAR_USER_AGENT not set in environment")
    return ua


# Paths
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"

# ─── LLM settings ─────────────────────────────────────────────────────────────
# Every LLM call goes through the Claude Code CLI (`claude -p`), which uses the
# login of whoever runs the backend. No API key is read by this project.
AGENT_MODEL = os.getenv("DD_AGENT_MODEL", "claude-opus-5-5")
SYNTHESIS_MODEL = os.getenv("DD_SYNTHESIS_MODEL", "claude-opus-5-5")

# Optional explicit path to the `claude` binary (otherwise found on PATH).
CLAUDE_CLI_PATH = os.getenv("CLAUDE_CLI_PATH", "")

# Wall-clock limits per CLI call, in seconds.
CLAUDE_GATHER_TIMEOUT = int(os.getenv("DD_GATHER_TIMEOUT", "900"))
CLAUDE_ANALYZE_TIMEOUT = int(os.getenv("DD_ANALYZE_TIMEOUT", "600"))
CLAUDE_SYNTHESIS_TIMEOUT = int(os.getenv("DD_SYNTHESIS_TIMEOUT", "900"))

# ─── Caching ──────────────────────────────────────────────────────────────────
# Off by default: every run gathers fresh data and makes fresh LLM calls.
# Set DD_USE_CACHE=1 to bring back disk caching (data sources, agent reports, memos).
USE_CACHE = _env_flag("DD_USE_CACHE", default=False)


def cache_enabled() -> bool:
    """Return True when disk caching is switched on (read at call time)."""
    return USE_CACHE
