"""
Tests for the Supervisor agent.

Covers: plan validation, completion check, report validation, budget enforcement.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from src.agents.supervisor import (
    supervisor_validate_plan,
    supervisor_check_complete,
    supervisor_validate_report,
    _wall_clock_exceeded,
)
from src.schemas import (
    SubQuestion,
    Finding,
    BudgetConfig,
    BudgetUsage,
    SupervisorValidatePlanInput,
    SupervisorCheckCompleteInput,
    SupervisorValidateReportInput,
)


class TestSupervisorValidatePlan:
    def test_approves_valid_plan(self):
        """Test that supervisor approves a plan within budget."""
        subquestions = [
            SubQuestion(id=f"sq_{i}", text=f"Question {i}", status="pending")
            for i in range(1, 4)
        ]
        budget_config = BudgetConfig(max_subquestions=5)
        budget_usage = BudgetUsage()

        plan_input = SupervisorValidatePlanInput(
            subquestions=subquestions,
            budget_config=budget_config,
            budget_usage=budget_usage,
        )

        output, trace = supervisor_validate_plan(plan_input)

        assert output.approved
        assert len(output.trimmed_subquestions) == 3
        assert trace.status == "success"

    def test_trims_excess_subquestions(self):
        """Test that supervisor trims subquestions exceeding budget."""
        subquestions = [
            SubQuestion(id=f"sq_{i}", text=f"Question {i}", status="pending")
            for i in range(1, 8)  # 7 questions
        ]
        budget_config = BudgetConfig(max_subquestions=3)
        budget_usage = BudgetUsage()

        plan_input = SupervisorValidatePlanInput(
            subquestions=subquestions,
            budget_config=budget_config,
            budget_usage=budget_usage,
        )

        output, trace = supervisor_validate_plan(plan_input)

        assert output.approved
        assert len(output.trimmed_subquestions) == 3
        assert "Trimmed" in output.reason

    def test_rejects_empty_plan(self):
        """Test that supervisor rejects an empty plan."""
        plan_input = SupervisorValidatePlanInput(
            subquestions=[],
            budget_config=BudgetConfig(),
            budget_usage=BudgetUsage(),
        )

        output, trace = supervisor_validate_plan(plan_input)

        assert not output.approved
        assert trace.status == "error"


class TestSupervisorCheckComplete:
    def test_routes_to_next_pending(self):
        """Test routing to next pending subquestion."""
        subquestions = [
            SubQuestion(id="sq_1", text="Q1", status="complete"),
            SubQuestion(id="sq_2", text="Q2", status="pending"),
            SubQuestion(id="sq_3", text="Q3", status="pending"),
        ]
        budget_config = BudgetConfig(max_searches_per_subquestion=3)
        budget_usage = BudgetUsage()
        now = datetime.now(timezone.utc).isoformat()

        check_input = SupervisorCheckCompleteInput(
            subquestions=subquestions,
            findings={"sq_1": []},
            budget_config=budget_config,
            budget_usage=budget_usage,
            created_at=now,
        )

        output, trace = supervisor_check_complete(check_input)

        assert not output.all_complete
        assert output.next_subquestion_id == "sq_2"

    def test_signals_complete_when_all_done(self):
        """Test that all complete returns all_complete=True."""
        subquestions = [
            SubQuestion(id=f"sq_{i}", text=f"Q{i}", status="complete")
            for i in range(1, 4)
        ]
        budget_config = BudgetConfig()
        budget_usage = BudgetUsage()
        now = datetime.now(timezone.utc).isoformat()

        check_input = SupervisorCheckCompleteInput(
            subquestions=subquestions,
            findings={f"sq_{i}": [] for i in range(1, 4)},
            budget_config=budget_config,
            budget_usage=budget_usage,
            created_at=now,
        )

        output, trace = supervisor_check_complete(check_input)

        assert output.all_complete
        assert output.next_subquestion_id is None

    def test_wall_clock_timeout(self):
        """Test that wall clock timeout triggers completion."""
        subquestions = [
            SubQuestion(id="sq_1", text="Q1", status="pending"),
        ]
        budget_config = BudgetConfig(wall_clock_timeout_seconds=1)
        budget_usage = BudgetUsage()

        # Set created_at to 10 seconds ago
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

        check_input = SupervisorCheckCompleteInput(
            subquestions=subquestions,
            findings={},
            budget_config=budget_config,
            budget_usage=budget_usage,
            created_at=old_time,
        )

        output, trace = supervisor_check_complete(check_input)

        assert output.all_complete
        assert output.budget_exhausted
        assert "timeout" in output.reason.lower()

    def test_token_budget_exhausted(self):
        """Test that token budget exhaustion triggers completion."""
        subquestions = [
            SubQuestion(id="sq_1", text="Q1", status="pending"),
        ]
        budget_config = BudgetConfig(max_total_tokens=100)
        budget_usage = BudgetUsage(total_tokens_used=200)  # Exceeds limit
        now = datetime.now(timezone.utc).isoformat()

        check_input = SupervisorCheckCompleteInput(
            subquestions=subquestions,
            findings={},
            budget_config=budget_config,
            budget_usage=budget_usage,
            created_at=now,
        )

        output, trace = supervisor_check_complete(check_input)

        assert output.all_complete
        assert output.budget_exhausted
        assert "token" in output.reason.lower()

    def test_skips_subquestion_with_exhausted_search_budget(self):
        """Test that subquestion with exhausted search budget is skipped."""
        subquestions = [
            SubQuestion(id="sq_1", text="Q1", status="pending"),
        ]
        budget_config = BudgetConfig(max_searches_per_subquestion=2)
        budget_usage = BudgetUsage(
            searches_per_subquestion={"sq_1": 2}  # Already did max searches
        )
        now = datetime.now(timezone.utc).isoformat()

        check_input = SupervisorCheckCompleteInput(
            subquestions=subquestions,
            findings={},
            budget_config=budget_config,
            budget_usage=budget_usage,
            created_at=now,
        )

        output, trace = supervisor_check_complete(check_input)

        # sq_1 has exhausted its budget, so we should be complete
        assert output.all_complete


class TestSupervisorValidateReport:
    def test_approves_report_with_citations(self):
        """Test that report with citations is approved."""
        report = """
        # Research Report
        The main finding is X [Source: https://example.com/1].
        We also found Y [Source: https://example.com/2].
        """

        report_input = SupervisorValidateReportInput(
            report=report,
            findings={"sq_1": []},
        )

        output, trace = supervisor_validate_report(report_input)

        assert output.approved
        assert output.citation_count >= 2

    def test_approves_report_with_url_citations(self):
        """Test that report with bare URL citations is approved."""
        # Report must be >100 chars to pass the length check
        report = (
            "This research report examines the topic thoroughly. "
            "Key findings can be found at https://example.com/1 "
            "and additional sources at https://example.com/2 confirm these results."
        )

        report_input = SupervisorValidateReportInput(
            report=report,
            findings={},
        )

        output, trace = supervisor_validate_report(report_input)

        assert output.approved
        assert output.citation_count >= 2

    def test_rejects_too_short_report(self):
        """Test that a very short report is rejected."""
        report = "Short."

        report_input = SupervisorValidateReportInput(
            report=report,
            findings={},
        )

        output, trace = supervisor_validate_report(report_input)

        assert not output.approved
        assert "short" in output.reason.lower() or "100" in output.reason

    def test_accepts_report_without_citations(self):
        """Test that a normal-length report without URLs is still accepted."""
        report = "A" * 200  # Long enough but no citations

        report_input = SupervisorValidateReportInput(
            report=report,
            findings={},
        )

        output, trace = supervisor_validate_report(report_input)

        # Should be accepted (citations optional — just noted)
        assert output.approved


class TestWallClockCheck:
    def test_not_exceeded(self):
        now = datetime.now(timezone.utc).isoformat()
        assert not _wall_clock_exceeded(now, 300)

    def test_exceeded(self):
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=400)).isoformat()
        assert _wall_clock_exceeded(old_time, 300)

    def test_handles_naive_datetime(self):
        """Test that naive datetimes are handled gracefully."""
        naive_time = datetime.utcnow().isoformat()  # No timezone info
        result = _wall_clock_exceeded(naive_time, 300)
        assert isinstance(result, bool)
