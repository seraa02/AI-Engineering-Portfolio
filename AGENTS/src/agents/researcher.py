"""
Researcher Agent — researches individual sub-questions via Tavily search.

Every finding contains: claim, source_url, source_snippet, retrieval_timestamp.
Handles budget enforcement per sub-question (max_searches_per_subquestion).
Uses claude-haiku-4-5 to synthesize findings into structured claims.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, TYPE_CHECKING

import anthropic

from src.schemas import (
    ResearcherInput,
    ResearcherOutput,
    Finding,
    TraceEntry,
    utc_now,
)
from src.tools.search import search_tavily

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a research assistant. Given a sub-question and raw search results, 
extract specific factual claims with their sources.

Rules:
1. Only extract claims that are directly supported by the search results
2. Each claim must reference a specific source URL
3. Be precise and factual — no speculation
4. Output ONLY valid JSON

Output format:
{
    "claims": [
        {
            "claim": "specific factual statement",
            "source_url": "https://...",
            "source_snippet": "relevant excerpt from source"
        }
    ]
}
"""


def run_researcher(
    researcher_input: ResearcherInput,
    anthropic_client: anthropic.Anthropic,
    tavily_client,
    model: str = "claude-haiku-4-5",
    searches_already_done: int = 0,
) -> tuple[ResearcherOutput, TraceEntry]:
    """
    Run the Researcher agent for a single sub-question.
    
    Args:
        researcher_input: Typed input with sub-question and budget
        anthropic_client: Initialized Anthropic client
        tavily_client: Initialized TavilyClient
        model: Model to use
        searches_already_done: Number of searches already performed for this subquestion
    
    Returns:
        Tuple of (ResearcherOutput, TraceEntry)
    """
    start_time = time.time()
    timestamp = utc_now()
    input_dict = researcher_input.model_dump()

    subquestion = researcher_input.subquestion
    max_searches = researcher_input.max_searches
    tool_calls_record: list[dict] = []
    all_raw_findings: list[Finding] = []
    searches_performed = 0

    try:
        # Determine how many more searches we can do
        remaining_searches = max_searches - searches_already_done

        if remaining_searches <= 0:
            logger.info(
                "Budget exhausted for subquestion %s (already did %d searches)",
                subquestion.id,
                searches_already_done,
            )
            # Return what we have (possibly empty)
            output = ResearcherOutput(
                subquestion_id=subquestion.id,
                findings=[
                    Finding(
                        claim="Search budget exhausted for this sub-question",
                        source_url="",
                        source_snippet="",
                        retrieval_timestamp=utc_now(),
                    )
                ],
                searches_performed=0,
            )
            trace_entry = TraceEntry(
                agent="researcher",
                input=input_dict,
                output=output.model_dump(),
                tool_calls=[],
                token_usage={"input_tokens": 0, "output_tokens": 0},
                latency_ms=(time.time() - start_time) * 1000,
                status="success",
                timestamp=timestamp,
            )
            return output, trace_entry

        # Perform Tavily searches
        # We generate query variations using the sub-question text directly
        queries = _generate_search_queries(subquestion.text, min(remaining_searches, max_searches))

        for query in queries:
            if searches_performed >= remaining_searches:
                break

            logger.info("Searching Tavily: query='%s' for subquestion=%s", query, subquestion.id)

            tool_call_start = time.time()
            raw_findings = search_tavily(
                query=query,
                tavily_client=tavily_client,
                max_results=5,
            )
            tool_call_latency = (time.time() - tool_call_start) * 1000

            tool_calls_record.append({
                "tool": "tavily_search",
                "input": {"query": query},
                "output": {"findings_count": len(raw_findings)},
                "latency_ms": tool_call_latency,
            })

            all_raw_findings.extend(raw_findings)
            searches_performed += 1

        # If we got no real findings, return empty result
        if not all_raw_findings:
            output = ResearcherOutput(
                subquestion_id=subquestion.id,
                findings=[
                    Finding(
                        claim="No findings retrieved",
                        source_url="",
                        source_snippet="",
                        retrieval_timestamp=utc_now(),
                    )
                ],
                searches_performed=searches_performed,
            )
            trace_entry = TraceEntry(
                agent="researcher",
                input=input_dict,
                output=output.model_dump(),
                tool_calls=tool_calls_record,
                token_usage={"input_tokens": 0, "output_tokens": 0},
                latency_ms=(time.time() - start_time) * 1000,
                status="success",
                timestamp=timestamp,
            )
            return output, trace_entry

        # Use Claude to synthesize raw findings into structured claims
        findings_text = _format_findings_for_synthesis(all_raw_findings)
        synthesis_prompt = (
            f"Sub-question: {subquestion.text}\n\n"
            f"Search results:\n{findings_text}\n\n"
            f"Extract factual claims from these results. Output only JSON."
        )

        response = anthropic_client.messages.create(
            model=model,
            max_tokens=2048,
            system=RESEARCHER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": synthesis_prompt}],
        )

        content = response.content[0].text.strip()
        token_usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        # Parse synthesized claims
        try:
            parsed = json.loads(content)
            raw_claims = parsed.get("claims", [])
        except (json.JSONDecodeError, Exception):
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    raw_claims = parsed.get("claims", [])
                except Exception:
                    raw_claims = []
            else:
                raw_claims = []

        # Build Finding objects from synthesized claims
        structured_findings: list[Finding] = []
        for claim_dict in raw_claims:
            if not isinstance(claim_dict, dict):
                continue
            claim_text = claim_dict.get("claim", "").strip()
            source_url = claim_dict.get("source_url", "")
            source_snippet = claim_dict.get("source_snippet", "")

            if not claim_text:
                continue

            structured_findings.append(
                Finding(
                    claim=claim_text,
                    source_url=source_url,
                    source_snippet=source_snippet[:500] if source_snippet else "",
                    retrieval_timestamp=utc_now(),
                )
            )

        # Fall back to raw findings if synthesis produced nothing
        if not structured_findings:
            structured_findings = all_raw_findings

        output = ResearcherOutput(
            subquestion_id=subquestion.id,
            findings=structured_findings,
            searches_performed=searches_performed,
        )

        latency_ms = (time.time() - start_time) * 1000
        trace_entry = TraceEntry(
            agent="researcher",
            input=input_dict,
            output=output.model_dump(),
            tool_calls=tool_calls_record,
            token_usage=token_usage,
            latency_ms=latency_ms,
            status="success",
            timestamp=timestamp,
        )

        logger.info(
            "Researcher found %d findings for subquestion=%s in %.1fms",
            len(structured_findings),
            subquestion.id,
            latency_ms,
        )
        return output, trace_entry

    except Exception as exc:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("Researcher failed for subquestion %s: %s", subquestion.id, exc)

        # Return a failed finding rather than raising — researcher should be resilient
        failed_finding = Finding(
            claim=f"Research failed: {str(exc)[:200]}",
            source_url="",
            source_snippet="",
            retrieval_timestamp=utc_now(),
        )
        output = ResearcherOutput(
            subquestion_id=subquestion.id,
            findings=[failed_finding],
            searches_performed=searches_performed,
        )

        trace_entry = TraceEntry(
            agent="researcher",
            input=input_dict,
            output=output.model_dump(),
            tool_calls=tool_calls_record,
            token_usage={"input_tokens": 0, "output_tokens": 0},
            latency_ms=latency_ms,
            status="error",
            timestamp=timestamp,
        )
        return output, trace_entry


def _generate_search_queries(subquestion_text: str, count: int) -> list[str]:
    """
    Generate search query variations from a sub-question.
    Simple approach: use the question directly + variations.
    """
    queries = [subquestion_text]

    if count >= 2:
        # Add a more specific variant
        queries.append(f"{subquestion_text} research findings facts")

    if count >= 3:
        # Add a broader variant
        queries.append(f"{subquestion_text} overview explanation")

    return queries[:count]


def _format_findings_for_synthesis(findings: list[Finding]) -> str:
    """Format raw findings for Claude synthesis prompt."""
    parts = []
    for i, finding in enumerate(findings[:10], 1):  # Cap at 10 to avoid token overflow
        parts.append(
            f"[{i}] URL: {finding.source_url}\n"
            f"    Content: {finding.source_snippet[:300]}\n"
        )
    return "\n".join(parts)
