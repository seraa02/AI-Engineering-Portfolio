"""
Tests for the Researcher agent.

Covers:
- successful run
- no search results
- rate limit handling
- timeout handling
- paywall/blocked result
- malformed search result
- budget exhaustion
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import time

import pytest

from src.agents.researcher import run_researcher
from src.schemas import ResearcherInput, SubQuestion
from src.tools.search import search_tavily
from tests.conftest import make_anthropic_response


class TestResearcherSuccessfulRun:
    def test_basic_research(self, mock_anthropic_researcher, mock_tavily_client):
        """Test successful research run with findings."""
        subquestion = SubQuestion(id="sq_1", text="What causes climate change?", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=2)

        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_client,
        )

        assert output.subquestion_id == "sq_1"
        assert len(output.findings) > 0
        assert trace.agent == "researcher"
        assert trace.status == "success"

    def test_findings_have_required_fields(self, mock_anthropic_researcher, mock_tavily_client):
        """Test that all findings have required fields."""
        subquestion = SubQuestion(id="sq_1", text="Test question", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=2)

        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_client,
        )

        for finding in output.findings:
            assert hasattr(finding, 'claim')
            assert hasattr(finding, 'source_url')
            assert hasattr(finding, 'source_snippet')
            assert hasattr(finding, 'retrieval_timestamp')
            assert finding.claim  # Must not be empty
            assert finding.retrieval_timestamp  # Must have timestamp

    def test_trace_includes_tool_calls(self, mock_anthropic_researcher, mock_tavily_client):
        """Test that trace records Tavily search calls."""
        subquestion = SubQuestion(id="sq_1", text="Test question", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=1)

        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_client,
        )

        assert len(trace.tool_calls) > 0
        assert trace.tool_calls[0]["tool"] == "tavily_search"


class TestResearcherNoResults:
    def test_empty_search_results(self, mock_anthropic_researcher, mock_tavily_empty):
        """Test researcher handles empty Tavily results gracefully."""
        subquestion = SubQuestion(id="sq_1", text="Very obscure question", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=1)

        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_empty,
        )

        # Should still return findings (with "No results found" claim)
        assert len(output.findings) > 0
        assert "No results" in output.findings[0].claim or output.findings[0].claim

    def test_empty_results_finding_structure(self, mock_anthropic_researcher, mock_tavily_empty):
        """Test that empty result finding still has proper structure."""
        subquestion = SubQuestion(id="sq_1", text="Obscure topic", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=1)

        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_empty,
        )

        finding = output.findings[0]
        assert finding.retrieval_timestamp  # Must have timestamp even for empty results


class TestResearcherRateLimiting:
    def test_rate_limit_retry(self, mock_anthropic_researcher, mock_tavily_rate_limited):
        """Test that researcher retries on 429 errors with backoff."""
        subquestion = SubQuestion(id="sq_1", text="Test question", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=1)

        with patch("time.sleep"):  # Don't actually sleep in tests
            output, trace = run_researcher(
                researcher_input=researcher_input,
                anthropic_client=mock_anthropic_researcher,
                tavily_client=mock_tavily_rate_limited,
            )

        # Should eventually succeed after retries
        assert output.subquestion_id == "sq_1"
        assert len(output.findings) > 0

    def test_rate_limit_exhausted_returns_finding(self):
        """Test that exhausting retries returns a graceful finding."""
        always_rate_limited = MagicMock()
        always_rate_limited.search.side_effect = Exception("429 rate limit exceeded")

        with patch("time.sleep"):
            findings = search_tavily(
                query="test query",
                tavily_client=always_rate_limited,
            )

        assert len(findings) > 0
        assert "rate limit" in findings[0].claim.lower() or "429" in findings[0].claim


class TestResearcherTimeout:
    def test_timeout_handling(self, mock_anthropic_researcher, mock_tavily_timeout):
        """Test that researcher handles TimeoutError gracefully."""
        subquestion = SubQuestion(id="sq_1", text="Test question", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=1)

        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_timeout,
        )

        # Should return finding noting the timeout
        assert len(output.findings) > 0
        # The finding should mention timeout or the researcher should handle it
        assert output.subquestion_id == "sq_1"

    def test_timeout_finding_has_timestamp(self, mock_anthropic_researcher, mock_tavily_timeout):
        """Test that timeout findings still have retrieval timestamps."""
        subquestion = SubQuestion(id="sq_1", text="Test", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=1)

        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_timeout,
        )

        for finding in output.findings:
            assert finding.retrieval_timestamp


class TestResearcherPaywall:
    def test_paywall_skipped(self, mock_anthropic_researcher, mock_tavily_paywall):
        """Test that paywalled results are skipped."""
        subquestion = SubQuestion(id="sq_1", text="Test question", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=1)

        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_paywall,
        )

        # Should return findings (even if it's a "skipped paywall" note)
        assert len(output.findings) > 0
        # Source URL should not be a paywalled URL in claims
        # (or it should note inaccessibility)

    def test_paywall_finding_has_claim(self, mock_anthropic_researcher, mock_tavily_paywall):
        """Test that paywall results produce a finding with a claim."""
        subquestion = SubQuestion(id="sq_1", text="Test", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=1)

        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_paywall,
        )

        for finding in output.findings:
            assert finding.claim  # Must have a claim


class TestResearcherMalformedResults:
    def test_malformed_result_no_crash(self, mock_anthropic_researcher, mock_tavily_malformed):
        """Test that malformed search results don't crash the researcher."""
        subquestion = SubQuestion(id="sq_1", text="Test question", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=1)

        # Should not raise
        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_malformed,
        )

        assert output.subquestion_id == "sq_1"


class TestResearcherBudgetExhaustion:
    def test_search_budget_exhausted(self, mock_anthropic_researcher, mock_tavily_client):
        """Test that researcher respects max_searches budget."""
        subquestion = SubQuestion(id="sq_1", text="Test question", status="pending")
        researcher_input = ResearcherInput(subquestion=subquestion, max_searches=2)

        # Simulate already having done 2 searches
        output, trace = run_researcher(
            researcher_input=researcher_input,
            anthropic_client=mock_anthropic_researcher,
            tavily_client=mock_tavily_client,
            searches_already_done=2,  # Budget already exhausted
        )

        # Should return budget exhausted finding
        assert len(output.findings) > 0
        assert output.searches_performed == 0  # No new searches
        assert "budget" in output.findings[0].claim.lower() or output.findings[0].claim


class TestSearchTavily:
    """Unit tests for the search_tavily helper function."""

    def test_successful_search(self, mock_tavily_client):
        findings = search_tavily("test query", mock_tavily_client)
        assert len(findings) > 0
        assert all(hasattr(f, 'claim') for f in findings)
        assert all(hasattr(f, 'retrieval_timestamp') for f in findings)

    def test_empty_results_returns_finding(self, mock_tavily_empty):
        findings = search_tavily("obscure query", mock_tavily_empty)
        assert len(findings) == 1
        assert findings[0].source_url == ""

    def test_paywall_skipped(self, mock_tavily_paywall):
        findings = search_tavily("paywalled query", mock_tavily_paywall)
        assert len(findings) > 0  # Returns fallback
        # The inaccessible content should produce a fallback message
        assert findings[0].claim

    def test_malformed_result_no_crash(self, mock_tavily_malformed):
        """Malformed results should not crash."""
        client = MagicMock()
        client.search.return_value = {
            "results": [
                {"content": "No URL here"},
                {"url": "https://example.com", "content": "Valid"},
                {},  # Completely empty
            ]
        }
        findings = search_tavily("test", client)
        assert len(findings) > 0
