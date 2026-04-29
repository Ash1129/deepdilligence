"""Job listings scraper for gauging company hiring activity and culture."""

import logging
import re
from typing import Any

import requests
from bs4 import BeautifulSoup

from src.utils.cache import disk_cache

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DeepDiligence/1.0; research bot)",
    "Accept": "text/html,application/xhtml+xml",
}


@disk_cache(subfolder="jobs")
def scrape_careers_page(careers_url: str, timeout: int = 15) -> dict[str, Any]:
    """Scrape a company careers page for job listing data.

    Args:
        careers_url: URL of the company's careers/jobs page.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys: url, job_count, job_titles, departments, raw_text, error.
    """
    try:
        resp = requests.get(careers_url, headers=DEFAULT_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch careers page %s: %s", careers_url, e)
        return {
            "url": careers_url,
            "job_count": 0,
            "job_titles": [],
            "departments": [],
            "raw_text": None,
            "error": str(e),
        }

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove noise
    for tag in soup.find_all(["script", "style", "nav", "footer"]):
        tag.decompose()

    # Extract job titles — common patterns in careers pages
    job_titles = _extract_job_titles(soup)
    departments = _extract_departments(soup)

    body_text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    raw_text = "\n".join(lines[:500])  # Cap to prevent huge cache files

    return {
        "url": careers_url,
        "job_count": len(job_titles),
        "job_titles": job_titles[:100],  # Cap at 100
        "departments": departments,
        "raw_text": raw_text,
        "error": None,
    }


def _extract_job_titles(soup: BeautifulSoup) -> list[str]:
    """Heuristically extract job titles from a careers page."""
    titles = set()

    # Look for common job listing patterns: links with job-like text
    for link in soup.find_all("a", href=True):
        text = link.get_text(strip=True)
        if _looks_like_job_title(text):
            titles.add(text)

    # Look for headings that might be job titles
    for heading in soup.find_all(["h2", "h3", "h4"]):
        text = heading.get_text(strip=True)
        if _looks_like_job_title(text):
            titles.add(text)

    return sorted(titles)


def _looks_like_job_title(text: str) -> bool:
    """Check if text looks like a job title based on heuristics."""
    if not text or len(text) < 5 or len(text) > 120:
        return False

    job_keywords = [
        r"\bengineer\b", r"\bmanager\b", r"\banalyst\b", r"\bdesigner\b",
        r"\bdeveloper\b", r"\bdirector\b", r"\blead\b", r"\bspecialist\b",
        r"\bcoordinator\b", r"\bscientist\b", r"\barchitect\b", r"\bconsultant\b",
        r"\bintern\b", r"\bvp\b", r"\bhead of\b",
    ]
    text_lower = text.lower()
    return any(re.search(kw, text_lower) for kw in job_keywords)


def _extract_departments(soup: BeautifulSoup) -> list[str]:
    """Heuristically extract department/team names from a careers page."""
    departments = set()
    dept_keywords = [
        "engineering", "product", "design", "marketing", "sales",
        "operations", "finance", "legal", "human resources", "data",
        "security", "infrastructure", "customer", "support",
    ]

    for element in soup.find_all(["span", "div", "li", "option", "a"]):
        text = element.get_text(strip=True).lower()
        if text and len(text) < 50:
            for kw in dept_keywords:
                if kw in text:
                    departments.add(element.get_text(strip=True))
                    break

    return sorted(departments)
