"""General web scraper for fetching and cleaning text from URLs."""

import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.utils.cache import disk_cache

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DeepDiligence/1.0; research bot)",
    "Accept": "text/html,application/xhtml+xml",
}

# Tags that typically contain noise rather than content
NOISE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"}


@disk_cache(subfolder="web")
def scrape_url(url: str, timeout: int = 15) -> dict[str, Optional[str]]:
    """Fetch a URL and return cleaned text content.

    Args:
        url: The URL to scrape.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys: url, title, text, error.
        On failure, text is None and error contains the reason.
    """
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch %s: %s", url, e)
        return {"url": url, "title": None, "text": None, "error": str(e)}

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove noise elements
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else None

    # Extract main content — prefer <main> or <article> if present
    main = soup.find("main") or soup.find("article") or soup.find("body")
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    return {"url": url, "title": title, "text": cleaned, "error": None}
