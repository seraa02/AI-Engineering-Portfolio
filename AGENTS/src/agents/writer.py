"""
Writer Agent — produces final report with citations.

Uses claude-sonnet-4-6 (higher quality) for the final report generation.
Report must include inline citations referencing source URLs.
"""

from __future__ import annotations

import json
import logging
import time
import re
from typing import TYPE_CHECKING

import anthropic

from src.schemas import (
    WriterInput,
    WriterOutput,
    TraceEntry,
    utc_now,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are a research report writer. Given a research question, sub-questions, 
and research findings, produce a comprehensive, well-structured report.

Rules:
1. Structure the report with clear sections and headers
2. EVERY factual claim must have an inline citation in the format [Source: URL]
3. If a finding has no URL, cite it as [Source: internal research]
4. Write in an objective, academic tone
5. Include a References section at the end listing all cited URLs
6. The report should be comprehensive but concise (aim for 500-1500 words)
7. Do NOT make up facts — only use what's in the findings

Report structure:
- Executive Summary
- [Sections based on sub-questions]
- Conclusion
- References
"""


def run_writer(
    writer_input: WriterInput,
    anthropic_client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
) -> tuple[WriterOutput, TraceEntry]:
    """
    Run the Writer agent to produce a final cited report.
    
    Args:
        writer_input: Typed input with question, subquestions, and findings
        anthropic_client: Initialized Anthropic client
        model: Model to use (default: claude-sonnet-4-6)
    
    Returns:
        Tuple of (WriterOutput, TraceEntry)
    """
    start_time = time.time()
    timestamp = utc_now()
    input_dict = {
        "question": writer_input.question,
        "subquestion_count": len(writer_input.subquestions),
        "finding_count": sum(len(f) for f in writer_input.findings.values()),
    }

    try:
        # Format findings for the writer prompt
        formatted_findings = _format_findings_for_report(
            writer_input.subquestions,
            writer_input.findings,
        )

        user_message = (
            f"Research Question: {writer_input.question}\n\n"
            f"Sub-questions researched:\n"
            + "\n".join(
                f"- {sq.text}" for sq in writer_input.subquestions
            )
            + f"\n\nResearch Findings:\n{formatted_findings}\n\n"
            f"Write a comprehensive research report with inline citations."
        )

        response = anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            system=WRITER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        report_text = response.content[0].text.strip()
        token_usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        # Count citations in report
        citation_count = _count_citations(report_text)

        # Ensure at least one citation exists
        if citation_count == 0:
            # Add a note if writer failed to include citations
            report_text += "\n\n[Note: Citations could not be automatically embedded in this report.]"
            citation_count = 0

        output = WriterOutput(report=report_text, citation_count=citation_count)

        latency_ms = (time.time() - start_time) * 1000
        trace_entry = TraceEntry(
            agent="writer",
            input=input_dict,
            output={"report_length": len(report_text), "citation_count": citation_count},
            tool_calls=[],
            token_usage=token_usage,
            latency_ms=latency_ms,
            status="success",
            timestamp=timestamp,
        )

        logger.info(
            "Writer produced report: %d chars, %d citations in %.1fms",
            len(report_text),
            citation_count,
            latency_ms,
        )
        return output, trace_entry

    except Exception as exc:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("Writer failed: %s", exc)

        trace_entry = TraceEntry(
            agent="writer",
            input=input_dict,
            output={"error": str(exc)},
            tool_calls=[],
            token_usage={"input_tokens": 0, "output_tokens": 0},
            latency_ms=latency_ms,
            status="error",
            timestamp=timestamp,
        )
        raise


def _format_findings_for_report(subquestions, findings_map) -> str:
    """Format all findings organized by sub-question for the writer prompt."""
    sections = []

    for sq in subquestions:
        sq_findings = findings_map.get(sq.id, [])
        section_parts = [f"Sub-question: {sq.text}"]

        if not sq_findings:
            section_parts.append("  [No findings available]")
        else:
            for i, finding in enumerate(sq_findings, 1):
                url_part = f" (source: {finding.source_url})" if finding.source_url else ""
                section_parts.append(f"  Finding {i}: {finding.claim}{url_part}")
                if finding.source_snippet:
                    section_parts.append(f"    Snippet: {finding.source_snippet[:200]}")

        sections.append("\n".join(section_parts))

    return "\n\n".join(sections)


def _count_citations(report_text: str) -> int:
    """Count inline citations in the report."""
    # Count [Source: ...] patterns
    source_citations = len(re.findall(r'\[Source:', report_text))
    # Also count numbered references like [1], [2], etc.
    numbered_citations = len(re.findall(r'\[\d+\]', report_text))
    # Count http URLs as citations
    url_citations = len(re.findall(r'https?://\S+', report_text))

    return max(source_citations, numbered_citations, url_citations)
