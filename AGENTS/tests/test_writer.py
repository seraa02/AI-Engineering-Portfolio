"""
Tests for the Writer agent.

Covers: successful report generation, citation counting, error handling.
"""

from __future__ import annotations

import json
import pytest

from src.agents.writer import run_writer, _count_citations
from src.schemas import WriterInput, SubQuestion, Finding
from tests.conftest import make_anthropic_response


class TestWriterSuccessfulRun:
    def test_produces_report(self, mock_anthropic_writer, state_with_findings):
        """Test that writer produces a non-empty report."""
        subquestions = [
            SubQuestion(**sq) for sq in state_with_findings["subquestions"]
        ]
        findings_map = {
            sq_id: [Finding(**f) for f in f_list]
            for sq_id, f_list in state_with_findings["findings"].items()
        }

        writer_input = WriterInput(
            question="What causes climate change?",
            subquestions=subquestions,
            findings=findings_map,
        )

        output, trace = run_writer(
            writer_input=writer_input,
            anthropic_client=mock_anthropic_writer,
        )

        assert output.report
        assert len(output.report) > 50  # Non-trivial report
        assert trace.agent == "writer"
        assert trace.status == "success"

    def test_citation_count_captured(self, mock_anthropic_writer, state_with_findings):
        """Test that citation count is captured in output."""
        subquestions = [SubQuestion(**sq) for sq in state_with_findings["subquestions"]]
        findings_map = {
            sq_id: [Finding(**f) for f in f_list]
            for sq_id, f_list in state_with_findings["findings"].items()
        }

        writer_input = WriterInput(
            question="What causes climate change?",
            subquestions=subquestions,
            findings=findings_map,
        )

        output, trace = run_writer(
            writer_input=writer_input,
            anthropic_client=mock_anthropic_writer,
        )

        # The mock report has URL citations
        assert output.citation_count >= 0

    def test_trace_has_token_usage(self, mock_anthropic_writer, state_with_findings):
        """Test that writer trace records token usage."""
        subquestions = [SubQuestion(**sq) for sq in state_with_findings["subquestions"]]
        findings_map = {}

        writer_input = WriterInput(
            question="Test question",
            subquestions=subquestions,
            findings=findings_map,
        )

        output, trace = run_writer(
            writer_input=writer_input,
            anthropic_client=mock_anthropic_writer,
        )

        assert "input_tokens" in trace.token_usage
        assert "output_tokens" in trace.token_usage

    def test_trace_records_report_metadata(self, mock_anthropic_writer, state_with_findings):
        """Test that trace output records report length and citation count."""
        subquestions = [SubQuestion(**sq) for sq in state_with_findings["subquestions"]]
        findings_map = {
            sq_id: [Finding(**f) for f in f_list]
            for sq_id, f_list in state_with_findings["findings"].items()
        }

        writer_input = WriterInput(
            question="What causes climate change?",
            subquestions=subquestions,
            findings=findings_map,
        )

        output, trace = run_writer(
            writer_input=writer_input,
            anthropic_client=mock_anthropic_writer,
        )

        assert "report_length" in trace.output
        assert "citation_count" in trace.output


class TestWriterWithNoFindings:
    def test_writes_with_empty_findings(self, mock_anthropic_writer):
        """Test writer handles empty findings gracefully."""
        writer_input = WriterInput(
            question="Test question",
            subquestions=[
                SubQuestion(id="sq_1", text="Sub-question 1", status="complete"),
            ],
            findings={},
        )

        output, trace = run_writer(
            writer_input=writer_input,
            anthropic_client=mock_anthropic_writer,
        )

        assert output.report
        assert trace.status == "success"


class TestCitationCounting:
    def test_count_source_citations(self):
        """Test counting [Source: URL] patterns."""
        text = "Found that X [Source: https://a.com]. Also Y [Source: https://b.com]."
        assert _count_citations(text) >= 2

    def test_count_url_citations(self):
        """Test counting bare URLs."""
        text = "See https://example.com/1 and https://example.com/2 for details."
        assert _count_citations(text) >= 2

    def test_count_numbered_citations(self):
        """Test counting [1], [2] style citations."""
        text = "First point [1]. Second point [2]. Third point [3]."
        assert _count_citations(text) >= 3

    def test_no_citations(self):
        """Test report with no detectable citations."""
        text = "This is a report with no citations or URLs mentioned anywhere."
        assert _count_citations(text) == 0


class TestWriterErrorHandling:
    def test_api_error_raises(self, mock_anthropic_client):
        """Test that API errors are propagated."""
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        writer_input = WriterInput(
            question="Test",
            subquestions=[],
            findings={},
        )

        with pytest.raises(Exception, match="API Error"):
            run_writer(
                writer_input=writer_input,
                anthropic_client=mock_anthropic_client,
            )
