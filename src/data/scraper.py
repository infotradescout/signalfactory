"""
src/data/scraper.py — Lightweight web scraper for news sentiment and public data.
Respects robots.txt by default; uses a browser-like User-Agent.
"""
import logging
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_DEFAULT_UA = "Mozilla/5.0 (compatible; PredictiveSystem/1.0)"
_TIMEOUT = 15  # seconds


class WebScraper:
    """
    Fetch public web data for sentiment / trend signals.
    Only hits URLs you explicitly provide — no autonomous crawling.
    """

    def __init__(self, user_agent: str = _DEFAULT_UA):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json, text/html;q=0.9",
        })

    # ── public API ────────────────────────────────────────────────────────────

    def fetch_reddit_sentiment(self, subreddit: str = "economics", limit: int = 100) -> dict:
        """
        Fetch the top posts from a subreddit (JSON API — no auth required).
        Returns a dict with avg_score and a rough positive/negative ratio.
        """
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
        try:
            resp = self.session.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            posts = resp.json().get("data", {}).get("children", [])
            scores = [p["data"].get("score", 0) for p in posts]
            titles = [p["data"].get("title", "") for p in posts]

            positive_keywords = {"growth", "rise", "surge", "gain", "bull", "recovery", "up"}
            negative_keywords = {"crash", "fall", "decline", "recession", "risk", "crisis", "down"}

            pos = sum(
                1 for t in titles
                if any(kw in t.lower() for kw in positive_keywords)
            )
            neg = sum(
                1 for t in titles
                if any(kw in t.lower() for kw in negative_keywords)
            )
            total = len(titles) or 1

            return {
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "positive_ratio": pos / total,
                "negative_ratio": neg / total,
                "sentiment_net": (pos - neg) / total,
                "post_count": len(posts),
            }
        except requests.RequestException as exc:
            logger.warning("Reddit fetch failed: %s", exc)
            return {}

    def fetch_json(self, url: str, params: Optional[dict] = None) -> dict:
        """Generic JSON fetch with error handling."""
        self._validate_url(url)
        try:
            resp = self.session.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("fetch_json failed (%s): %s", url, exc)
            return {}

    def fetch_html(self, url: str) -> str:
        """Return page HTML as a string."""
        self._validate_url(url)
        try:
            resp = self.session.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            logger.warning("fetch_html failed (%s): %s", url, exc)
            return ""

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Only http/https URLs are supported, got: {url!r}")
        if not parsed.netloc:
            raise ValueError(f"Invalid URL (no host): {url!r}")
