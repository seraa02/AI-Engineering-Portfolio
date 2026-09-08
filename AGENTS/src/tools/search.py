"""
Tavily search wrapper with comprehensive error handling.

Handles:
- Empty results: returns Finding with claim="No results found"
- HTTP 429 (rate limit): exponential backoff (1s, 2s, 4s) up to 3 attempts
- Timeout: catches TimeoutError, records as failed finding
- Paywall (non-200, no snippet): skips and records warning
- Malformed result (missing fields): uses getattr with defaults, never crashes
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from src.schemas import Finding

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 1  # seconds


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_empty_finding(reason: str = "No results found") -> Finding:
    return Finding(
        claim=reason,
        source_url="",
        source_snippet="",
        retrieval_timestamp=_utc_now(),
    )


def _fetch_page_text(url: str, timeout: float = 5.0, max_chars: int = 2000) -> str:
    """Fetch a URL and extract plain text content.

    The PDF spec requires fetching and extracting the full page rather than
    trusting search snippets, which are often truncated or SEO-optimised.
    Falls back silently on any network / parse error so search never crashes.
    """
    import html
    import re
    import urllib.request

    if not url or not url.startswith("http"):
        return ""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)",
                "Accept": "text/html,application/xhtml+xml,text/plain",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(100_000).decode("utf-8", errors="replace")
        # Strip script/style blocks
        raw = re.sub(
            r"<(script|style)[^>]*>.*?</(script|style)>",
            " ", raw, flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", " ", raw)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


def _parse_tavily_result(result: dict) -> Optional[Finding]:
    """
    Safely parse a single Tavily result dict into a Finding.
    Uses getattr-style .get() with defaults — never crashes on missing fields.
    """
    url = result.get("url", "")
    content = result.get("content", "")
    title = result.get("title", "")

    # Paywall detection: no content snippet
    if not content and not title:
        logger.warning("Skipping result with no content (possible paywall): url=%s", url)
        return None

    # Build a claim from title + content (truncated)
    snippet = content[:500] if content else title[:200]
    claim = f"{title}: {snippet}" if title and content else (title or snippet)

    return Finding(
        claim=claim.strip(),
        source_url=url,
        source_snippet=snippet.strip(),
        retrieval_timestamp=_utc_now(),
    )


def search_tavily(
    query: str,
    tavily_client,
    max_results: int = 5,
) -> list[Finding]:
    """
    Execute a Tavily search synchronously with retry logic.
    
    Args:
        query: Search query string
        tavily_client: Initialized TavilyClient instance
        max_results: Maximum number of results to fetch
    
    Returns:
        List of Finding objects (never empty — returns at least one with "No results found")
    """
    attempt = 0
    last_error = None

    while attempt < MAX_RETRIES:
        try:
            response = tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )

            results = response.get("results", []) if isinstance(response, dict) else []

            if not results:
                logger.info("Tavily returned no results for query: %s", query)
                return [_make_empty_finding("No results found")]

            findings = []
            for result in results:
                finding = _parse_tavily_result(result)
                if finding is not None:
                    findings.append(finding)

            if not findings:
                # All results were paywalled/empty
                return [_make_empty_finding("All results were inaccessible (paywall or empty)")]

            # Enrich each finding by fetching the full page text.
            # This replaces the Tavily snippet (which is often truncated/SEO text)
            # with the actual page content, as required by the spec.
            enriched = []
            for f in findings:
                full_text = _fetch_page_text(f.source_url) if f.source_url else ""
                if full_text and len(full_text) > len(f.source_snippet):
                    f = Finding(
                        claim=f.claim,
                        source_url=f.source_url,
                        source_snippet=full_text[:500],
                        retrieval_timestamp=f.retrieval_timestamp,
                    )
                enriched.append(f)
            return enriched

        except Exception as exc:
            exc_str = str(exc)
            last_error = exc_str

            # Rate limit: HTTP 429
            if "429" in exc_str or "rate limit" in exc_str.lower() or "too many requests" in exc_str.lower():
                wait = BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Tavily rate limit hit (attempt %d/%d). Sleeping %ds. Error: %s",
                    attempt + 1, MAX_RETRIES, wait, exc_str
                )
                time.sleep(wait)
                attempt += 1
                continue

            # Timeout
            if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in exc_str.lower():
                logger.error("Tavily search timed out for query: %s", query)
                return [_make_empty_finding(f"Search timed out: {exc_str}")]

            # Generic error — log and return empty finding
            logger.error("Tavily search error for query '%s': %s", query, exc_str)
            return [_make_empty_finding(f"Search error: {exc_str}")]

    # Exhausted retries (only for rate limit path)
    logger.error(
        "Tavily search failed after %d attempts for query '%s': %s",
        MAX_RETRIES, query, last_error
    )
    return [_make_empty_finding(f"Rate limit exceeded after {MAX_RETRIES} attempts")]
