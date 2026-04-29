"""Configuration management for DeepDiligence."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


def get_openai_api_key() -> str:
    """Return the OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    return key


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
CACHE_DIR = _project_root / "data" / "cache"
BENCHMARKS_DIR = _project_root / "data" / "benchmarks"

# LLM settings — gpt-5.4-mini for all agents and synthesis
AGENT_MODEL = "gpt-5.4-mini"
SYNTHESIS_MODEL = "gpt-5.4-mini"
