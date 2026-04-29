"""NewsAPI client for fetching recent news about a company."""

import logging
from datetime import datetime, timedelta
from typing import Any

import requests

from src.utils.cache import disk_cache
from src.utils.config import get_news_api_key

logger = logging.getLogger(__name__)

NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"


@disk_cache(subfolder="news")
def fetch_company_news(
    company_name: str,
    days_back: int = 30,
    page_size: int = 20,
    language: str = "en",
) -> list[dict[str, Any]]:
    """Fetch recent news articles about a company from NewsAPI.

    Args:
        company_name: Company name to search for.
        days_back: How many days back to search.
        page_size: Max number of articles to return.
        language: Language filter.

    Returns:
        List of article dicts with keys: title, description, url, source,
        published_at, content.
    """
    api_key = get_news_api_key()
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q": f'"{company_name}"',
        "from": from_date,
        "sortBy": "relevancy",
        "pageSize": page_size,
        "language": language,
        "apiKey": api_key,
    }

    resp = requests.get(NEWSAPI_EVERYTHING_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        logger.error("NewsAPI error: %s", data.get("message", "unknown"))
        return []

    articles = []
    for article in data.get("articles", []):
        articles.append({
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "url": article.get("url", ""),
            "source": article.get("source", {}).get("name", ""),
            "published_at": article.get("publishedAt", ""),
            "content": article.get("content", ""),
        })

    return articles
